#
# sympy_check
# I. Chuang <ichuang@mit.edu>
# 06-Mar-2014
#
#-----------------------------------------------------------------------------
# symbolic expression checker for edX custom response problems.
#
# Includes unit tests(!).  Run tests with "py.test sympy_check.py"
#
# Usage examples:
# 
# \edXincludepy{lib/sympy_check.py}
# \edXinline{$s_3 = $} \edXabox{type="custom" size=30 expect="m1 + m2 + m3" cfn=sympy_formula_check inline="1"}
#
# two possible accepted answers (alternate answer specified using options; note single quotes):
#
# \edXabox{type="custom" size=30 expect="6*p" options="altanswer='6*p-18*p*r'" math=1 cfn=sympy_formula_check math=1 inline="1"}
# 
# \edXinline{$p_{\rm ftnand,1} = $} 
# \edXabox{type="custom" 
#   size=30 
#   expect="3*36*p^2 + 72*p*e + 24*e^2" 
#   options="altanswer='3*36*p^2 + 24*e^2'"
#   cfn=sympy_formula_check 
#   math=1 inline="1"
# }
# 
# numerical value checking with tolerances:
# 
# \edXinline{$p_{th} = $} 
# \edXabox{type="custom" size=30 
#   expect="1/192" 
#   options="altanswer='1/120'!tolerance='0.0003'"
#   inline="1" 
#   cfn=sympy_formula_check 
#   hints='myhints'
# }
#
# vector / list checking:
#
# \edXinline{{\tt [IX,IZ,XI,ZI]}$\rightarrow$} \edXabox{type="custom" size=40 expect="[XI,ZI,IX,IZ]" noanswer=1 cfn=sympy_formula_check inline="1"}
#
# matrix checking:
#
# \edXinline{Density matrix $\rho'$ for Alice's state $|\psi'_{A,k}\>$:} 
# \edXabox{type="custom" size=60 expect="[[3/4,0],[0,1/4]]" cfn="sympy_formula_check" 
#   math="1"
#   inline="1"}
#
# \edXabox{type="custom" size=60 expect="[[a+f,c+h],[j+p,m+r]]" cfn="sympy_formula_check" math=1}
#
# checking of equations with non-commutative symbols:
#
# \item \edXinline{$B = $} \edXabox{type="custom" size=20 expect="h*A" cfn=sympy_check_nocomm inline="1"}
#
# checking quantum states (kets):
#
# \edXinline{$|s_1\> = $} 
# \edXabox{type="custom" size=60 
#   expect="a*|00> + a*exp(i*theta)* |01>+b*|11> + b* exp(i*theta)*|10>" 
#   cfn="sympy_check_quantum" 
#   inline="1"
#   math=1
#   preprocessorClassName="MathjaxPreprocessorForQM" preprocessorSrc="/static/mathjax_preprocessor_for_QM_H.js"
# }
#
#-----------------------------------------------------------------------------
# Theory:
#
# Everything is defined within a single central class, SympyFormulaChecker.
# This class can be instantiated with a number of options, e.g. defining symbols
# with special properties (assume certain variables are real, or do not commute,
# etc.).  
#
# The main entry point is sympy_formula_check(expect, ans, options)
#
# expect  = (correct) answer expected by instructor
# ans     = response given by student
# options = string specifying a variety of options, e.g. alternate answers accepted,
#           tolerances for comparison, and numerical sampling ranges for equality testing.
#
# The sympy formula checker uses python's sympy symbolic math package, to
# parse input, and also to represent mathematical input.  Vectors (lists) and matrices
# can be input and are represented properly.  Equations acting on matrices are
# legal.  
#
# Expression equality is first checked symbolically.  This can sometimes fail, however,
# due to complexity of possible expressions, and the inability of sympy to 
# simplify all possible expressions (e.g. complex trig).  Thus, a second step
# of equality testing is then tried, using random numerical sampling.
#
# The instructor can specify lists of variables and ranges for sampling the
# variables.  If such a sampling specification string is omitted, sympy_check
# will automatically generate samples, by taking all free variables, and 
# sampling them between 0 and 1 (exclusive).  This sampling range works
# remarkably well in a very wide range of use cases, but not always, so beware.
#
# Two global variables are defined, for convenience:
#
# sfc                 = an instance of SympyFormulaChecker, with default configuration
# sympy_formula_check = the sympy_formula_check entry point for sfc
#
#-----------------------------------------------------------------------------

import re
import numbers
import numpy
import sympy
import random
import copy
from sympy import Matrix, I
import sympy.physics.quantum
import sympy.physics.quantum.qubit

#-----------------------------------------------------------------------------

class SympyFormulaChecker(object):
    '''
    symbolic math expression checker based on sympy.

    Allows symbolic expression equality to be checked both symbolically, and
    using random numerical sampling.
    '''
    def __init__(self, extra_symbols=None, pre_parse_fn=None, pre_evaluator_fn=None, do_quantum=False, **kwargs):
        '''
        extra_symbols = extra math symbols (e.g. with special properties, like being real)
        pre_parse_fn = function executed on symbolic input before parsing with sympify
        pre_evaluator_fn = function executed by sympy_evaluator before evaluating to become numerical
        '''

        self.default_tolerance = '0.01%'

        self.special_vars = {
            'p':sympy.Symbol('p',real=True, positive=True),
            'g':sympy.Symbol('g',real=True),
        }

        self.varset = {#'p':sympy.Symbol('p'),
            #'g':sympy.Symbol('g'),
            'e':sympy.E,			# for exp
            'pi':sympy.pi,			# for pi
            'i':sympy.sympify('I'),		# lowercase i is also sqrt(-1)
            'X':sympy.sympify('Matrix([[0,1],[1,0]])'),
            'Y':sympy.sympify('Matrix([[0,-I],[I,0]])'),
            'Z':sympy.sympify('Matrix([[1,0],[0,-1]])'),
            'ZZ': sympy.Symbol('ZZ'),
        }
        self.varset.update(self.special_vars)

        self.extra_symbols = extra_symbols
        self.pre_parse_fn = pre_parse_fn
        self.pre_evaluator_fn = pre_evaluator_fn

        self.debug = False

        if do_quantum:
            def parse_ket(expr):
                return re.sub('\|([01]+?)>',r"qubit('\1')",expr.lower().replace('\n','').replace('\r',''))

            symtab = {'qubit':sympy.physics.quantum.qubit.Qubit,
                      'Ket':sympy.physics.quantum.state.Ket,
                      'bit':sympy.Function('bit'),
            }
            self.extra_symbols = symtab
            self.pre_parse_fn = parse_ket
            self.pre_evaluator_fn = sympy.physics.quantum.represent

        if self.extra_symbols is not None:
            self.special_vars.update(self.extra_symbols)
            self.varset.update(self.extra_symbols)

        self.update(kwargs)

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)
        
    @staticmethod
    def to_latex(x):
        xs = sympy.latex(x)
        if xs[0]=='$':
            return '[mathjax]%s[/mathjax]<br>' % (xs[1:-1])	# for sympy v6
        return '[mathjax]%s[/mathjax]<br>' % (xs)		# for sympy v7
        
    def parse_input(self, ein):

        if self.pre_parse_fn is not None:
            esub = self.pre_parse_fn(ein)
        else:
            esub = ein

        esub = re.sub('\[\s*\[([^\]]+)]\s*,\s*\[([^\]]+)\]\s*\]', 'Matrix([[\\1],[\\2]])', esub)	# 2-dim matrices
        esub = re.sub('\[\s*\[([^\]]+)]\s*,\s*\[([^\]]+)\],\s*\[([^\]]+)\]\s*\]', 'Matrix([[\\1],[\\2],[\\3]])', esub)	# 3-dim matrices
        esub = re.sub('\[\s*\[([^\]]+)]\s*,\s*\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\]\s*\]', 'Matrix([[\\1],[\\2],[\\3],[\\4]])', esub)	# 4-dim matrices
        esub = esub.replace('^','**')
    
        try:
            r = sympy.sympify(esub, locals=self.varset)
            return r
        except Exception as err:
            r = sympy.sympify('Matrix(%s)' % esub,locals=self.varset)
            return r
    
    #-----------------------------------------------------------------------------
    # formula comparison
    
    def sympy_evaluator(self, variables, functions, math_expr, case_sensitive=False):
        '''
        Return numerical expression for sympy symbolic expression "expr".
    
        variables = dict of variable name, numerical value.
    
        Use the "subs" method provided by sympy.
        '''
    
        # In sympy, Symbols with assumptions (like real=True) are different
        # from Symbols without them, even if they have the same name.
        #
        # In order to have nice expression simplification, we need to declare some
        # symbols as being real, e.g. g and p.  These are defined in the special_vars
        # dict.
        #
        # To be able to evaluate them with evalf() and subs(), we need to specify
        # those variables using the right Symbol object, i.e. those with the
        # assumptions specified.   
        #
        # Thus, for variables with names matching those in special_vars, we add 
        # additional entries to the variables dict, to make the symbols match.
    
        if self.pre_evaluator_fn is not None:
            math_expr = self.pre_evaluator_fn(math_expr)

        nvarset = {}
        for vname, vval in variables.items():
            nvarset[sympy.Symbol(vname)] = vval
            if vname in self.special_vars:
                nvarset[self.special_vars[vname]] = vval

        # evaluate "i" as complex root of minus unity
        nvarset[sympy.Symbol('i')] = 0 + 1j
    
        if isinstance(math_expr, str):		# tolerances are provided as strings - sympify them
            math_expr = sympy.sympify(math_expr)

        if type(math_expr)==list:
            try:
                ret = [ x.evalf(subs=nvarset) for x in math_expr ]
                return ret
            except Exception as err:
                raise Exception("[sympy_evaluator] Failed to evaluate expression %s err=%s" % (math_expr, err))

        try:
            ret = math_expr.evalf(subs=nvarset)
            return ret
        except Exception as err:
            raise Exception("[sympy_evaluator] Failed to evaluate expression %s err=%s" % (math_expr, err))
        
    
    @staticmethod
    def sympy_is_matrix(m):
        '''
        return True if m is a sympy matrix (mutable or immutable)
        '''
        if hasattr(sympy, 'ImmutableMatrix'):
            return isinstance(m, sympy.Matrix) or isinstance(m, sympy.ImmutableMatrix)
        return isinstance(m, sympy.Matrix)
    
    @staticmethod
    def is_list(x):
        '''
        return True if x is a list
        '''
        return type(x)==list

    def sympy_compare_with_tolerance(self, complex1, complex2, tolerance=None, relative_tolerance=False):
        """
        Compare complex1 to complex2 with maximum tolerance tol.
    
        If tolerance is type string, then it is counted as relative if it ends in %; otherwise, it is absolute.
    
         - complex1    :  student result (float complex number)
         - complex2    :  instructor result (float complex number)
         - tolerance   :  string representing a number or float
         - relative_tolerance: bool, used when`tolerance` is float to explicitly use passed tolerance as relative.
    
         Default tolerance of 1e-3% is added to compare two floats for
         near-equality (to handle machine representation errors).
         Default tolerance is relative, as the acceptable difference between two
         floats depends on the magnitude of the floats.
         (http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
         Examples:
            In [183]: 0.000016 - 1.6*10**-5
            Out[183]: -3.3881317890172014e-21
            In [212]: 1.9e24 - 1.9*10**24
            Out[212]: 268435456.0
        """
        def myabs(elem):
            if isinstance(elem, numpy.matrix):
                return numpy.sum(abs(elem))
            elif self.sympy_is_matrix(elem):
                return elem.norm()
            elif self.is_list(elem):
                return numpy.sum(map(abs, elem))
            return abs(elem)
    
        if tolerance is None:
            tolerance = self.default_tolerance
        if isinstance(tolerance, numbers.Number):
            tolerance = str(tolerance)
        if relative_tolerance:
            tolerance = tolerance * max(myabs(complex1), myabs(complex2))
        elif tolerance.endswith('%'):
            tolerance = sympy.sympify(tolerance[:-1]) * 0.01
            tolerance = tolerance * max(myabs(complex1), myabs(complex2))
        else:
            tolerance = self.sympy_evaluator(dict(), dict(), tolerance)
        if self.debug:
            print "c1=%s, c2=%s, tolerance=%s" %  (complex1, complex2, tolerance)

        # compare matrices

        if self.sympy_is_matrix(complex1) and not self.sympy_is_matrix(complex2):
            print "Oops, cannot compare sympy Matrix %s with non-sympy matrix %s" % (complex1, complex2)
            print "types = %s, %s" % (type(complex1), type(complex2))
            raise
    
        if self.sympy_is_matrix(complex2) and not self.sympy_is_matrix(complex1):
            print "Oops, cannot compare sympy Matrix %s with non-sympy matrix %s" % (complex1, complex2)
            print "types = %s, %s" % (type(complex1), type(complex2))
            raise
    
        try:
            if self.sympy_is_matrix(complex1) and self.sympy_is_matrix(complex2):
                if not (complex1.rows==complex2.rows):
                    print "Matrix sizes wrong, in comparison"
                    return False
                if not (complex1.cols==complex2.cols):
                    print "Matrix sizes wrong, in comparison"
                    return False
                diff = complex1 - complex2
                return diff.norm() <= tolerance            
        except Exception as err:
            print "failure in matrix comparison, complex1=%s, complex2=%s" % (complex1, complex2)
            print "err = ", err
            raise
    
        # compare lists

        if self.is_list(complex1) and not self.is_list(complex2):
            print "Oops, cannot compare list %s with non-list %s" % (complex1, complex2)
            print "types = %s, %s" % (type(complex1), type(complex2))
            raise

        if self.is_list(complex2) and not self.is_list(complex1):
            print "Oops, cannot compare list %s with non-list %s" % (complex2, complex1)
            print "types = %s, %s" % (type(complex1), type(complex2))
            raise

        try:
            if self.is_list(complex1) and self.is_list(complex2):
                if not (len(complex1)==len(complex2)):
                    print "list sizes wrong, in comparison"
                    return False
                diff = [x-y for x,y in zip(complex1, complex2)]
                return myabs(diff) <= tolerance            
        except Exception as err:
            print "failure in list comparison, complex1=%s, complex2=%s" % (complex1, complex2)
            print "err = ", err
            raise

        # compare numbers

        try:
            if sympy.mpmath.isinf(abs(complex1)) or sympy.mpmath.isinf(abs(complex2)):
                # If an input is infinite, we can end up with `abs(complex1-complex2)` and
                # `tolerance` both equal to infinity. Then, below we would have
                # `inf <= inf` which is a fail. Instead, compare directly.
                cmp = (complex1 == complex2)
                if isinstance(cmp, numpy.matrix):
                    return cmp.all()
                if self.sympy_is_matrix(cmp):
                    return cmp.all()
                return cmp
            else:
                # v1 and v2 are, in general, complex numbers:
                # there are some notes about backward compatibility issue: see responsetypes.get_staff_ans()).
                # return abs(complex1 - complex2) <= tolerance
                #
                # sum() used to handle matrix comparisons
                return numpy.sum(abs(complex1 - complex2)) <= tolerance
        except Exception as err:
            print "failure in comparison, complex1=%s, complex2=%s" % (complex1, complex2)
            print "err = ", err
            raise
    
    
    def sympy_make_samples(self, expr, rmin='0', rmax='0.999', nsamples=20):
        '''
        Make a "samples" sampling string, from a symbolic expression, for 
        random numerical checking.  Sample every variable between 0 and 1
        (exclusive).
        '''
        def gfs(ex):
            if self.sympy_is_matrix(ex):		# get free symbols from matrix elements one element at a time
                return sympy.flatten(map(gfs, list(ex)), 1)
            try:
                return ex.free_symbols	# constant-valued matrices will raise exception on free_symbols
            except:
                return []
        if isinstance(expr, list) and isinstance(expr[0], list):	# each expression is a list
            expr = sympy.flatten(expr, 1)	# merge lists
        if isinstance(expr, list):
            variables = sympy.flatten(map(gfs, expr), 1)
        else:
            variables = gfs(expr)
        variables = map(str, variables)
    
        if not variables:		# no free vaiables?  use a dummy one
            variables = 'p'
            
        nvar = len(variables)
        samples = ','.join(variables) + '@' + ','.join([rmin]*nvar) + ":" + ','.join([rmax]*nvar)
        samples += '#%d' % nsamples
        return samples
    
    
    def sympy_is_formula_equal(self, expected, given, samples=None, cs=True, tolerance='0.01', evalfun=None, cmpfun=None, debug=None, nsamples=20):
        '''
        Check for equality of two symbolic expressions.  Do this first using 
        symbolic comparison, then, failing that, try random numerical sampling.
    
        expected = instructor's expression
        given = student's expression
        samples = sample string for numerical checking (see below)
        cs = case_sensitive flag
        tolerance = tolerance specification string
        evalfun = function for doing evaluation (defaults to using evaluator from calc2)
        cmpfun = comparison function for testing equality (defaults to compare_with_tolerance)
        debug = flag for verbosity of debugging output
        
            format of samples:  <variables>@<lower_bounds>:<upper_bound>#<num_samples
    
            * variables    - a set of variables that are allowed as student input
            * lower_bounds - for every variable defined in variables, a lower
                             bound on the numerical tests to use for that variable
            * upper_bounds - for every variable defined in variables, an upper
                             bound on the numerical tests to use for that variable
        
        If samples is not specified, then sample every variable in the range [0,1)
        with nsamples samples chosen uniformly randomly.
    
        '''
        if (expected==given):
            return True
    
        if debug is None:
            debug = self.debug
        if evalfun is None:
            evalfun = self.sympy_evaluator
        if cmpfun is None:
            cmpfun = self.sympy_compare_with_tolerance
        if samples is None:
            samples = self.sympy_make_samples([expected, given])
            
        try:
            variables = samples.split('@')[0].split(',')
            numsamples = int(samples.split('@')[1].split('#')[1])
        except Exception as err:
            raise Exception("bad samples specification %s, cannot get variables or number of samples; err=%s" % (samples, err))
        
        def to_math_atom(sstr):
            '''
            Convert sample range atom to float or to matrix
            '''
            if '[' in sstr:
                try:
                    return numpy.matrix(sstr.replace('|',' '))
                except Exception as err:
                    raise Exception("Cannot generate matrix from %s; err=%s" % (sstr, err))
            elif 'j' in sstr:
                return complex(sstr)
            else:
                return float(sstr)
        
        try:
            sranges = zip(*map(lambda x: map(to_math_atom, x.split(",")),
                               samples.split('@')[1].split('#')[0].split(':')))
            ranges = dict(zip(variables, sranges))
        except Exception as err:
            raise Exception("bad samples specification %s, cannot get ranges; err=%s" % (samples, err))
    
        if debug:
            print "ranges = ", ranges
    
        for i in range(numsamples):
            vvariables = {}
            for var in ranges:
                value = random.uniform(*ranges[var])
                vvariables[str(var)] = value
            if debug:
                print "vvariables = ", vvariables
            try:
                instructor_result = evalfun(vvariables, dict(), expected, case_sensitive=cs)
            except Exception as err:
                #raise Exception("is_formula_eq: vvariables=%s, err=%s" % (vvariables, str(err)))
                #raise Exception("-- %s " % str(err))
                raise Exception("Error evaluating instructor result, expected=%s, vv=%s -- %s " % (expected, vvariables, str(err)))
            try:
                student_result = evalfun(vvariables, dict(), given, case_sensitive=cs)
            except Exception as err:
                if debug:
                    raise Exception("is_formula_eq: given=%s, vvariables=%s, err=%s" % (given, vvariables, str(err)))
                raise Exception("-- %s " % str(err))
                # raise Exception("Error evaluating your input, given=%s, vv=%s -- %s " % (given, vvariables, str(err)))
            #print "instructor=%s, student=%s" % (instructor_result, student_result)
            cfret = cmpfun(instructor_result, student_result, tolerance)
            if debug:
                print "comparison result = %s" % cfret
            if not cfret:
                return False
        return True
        
    #-----------------------------------------------------------------------------
    # sympy formula check
    
    def sympy_formula_check(self, expect, ans, options=None):
        '''
        expect and ans are math expression strings.
        Check for equality using random sampling.
    
        The strings are turned into math expressions using sympy.
    
        options should be like samples="m_I,m_J,I_z,J_z@1,1,1,1:20,20,20,20#50"!tolerance=0.3
        i.e. a sampling range for the equality testing, in the same
        format as used in formularesponse.
    
        options may also include altanswer, an alternate acceptable answer.  Example:
    
        options="samples='X,Y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'!altanswer='-Y*X'"

        if options has only_symbolic=1 then only symbolic equality is accepted.
    
        note that the different parts of the options string are to be spearated by a bang (!).
        '''
        samples = None
        tolerance = '0.1%'
        acceptable_answers = [expect]
        check_none = False
        only_symbolic = False
        if options is not None:
            for optstr in options.split('!'):
                if 'samples=' in optstr:
                    samples = eval(optstr.split('samples=')[1])
                elif 'tolerance=' in optstr:
                    tolerance = eval(optstr.split('tolerance=')[1])
                elif 'altanswer=' in optstr:
                    altanswer = eval(optstr.split('altanswer=')[1])
                    acceptable_answers.append(altanswer)
                elif 'check_none=' in optstr:
                    check_none = eval(optstr.split('check_none=')[1])
                elif 'only_symbolic=' in optstr:
                    only_symbolic = eval(optstr.split('only_symbolic=')[1])
    
        # fix escaping coming from edx capa
        def fix_escape(expr):
            gtch = chr(62)
            ampch = chr(38)
            return expr.replace(ampch + 'gt;', gtch)

        if check_none:
            ans = ans.lower()
            ok = ans in ['none', 'empty']
            return {'ok': ok, 'msg': ''}

        try:
            ans = self.parse_input(ans)
        except Exception as err:
            return {'ok': False, 'msg': 'Oops, your input could not be parsed, err=%s' % err}
    
        for acceptable_str in acceptable_answers:
            acceptable_str = fix_escape(acceptable_str)
            acceptable = self.parse_input(acceptable_str)
            try:
                if only_symbolic:
                    ok = (acceptable==ans)
                else:
                    if self.debug:
                        print "acceptable=%s, ans=%s, samples=%s" % (acceptable, ans, samples)
                        self.expect = acceptable
                        self.ans = ans
                    ok = self.sympy_is_formula_equal(acceptable, ans, samples, tolerance=tolerance)
                    if self.debug:
                        print "ok=%s" % ok
            except Exception as err:
                return {'ok': False, 'msg': "Sorry, could not evaluate your expression.  Error %s" % str(err)}
            if ok:
                return {'ok':ok, 'msg': ''}
    
        return {'ok':ok, 'msg': ''}
    
#-----------------------------------------------------------------------------
# global entry points
    
sfc = SympyFormulaChecker()
sympy_formula_check = sfc.sympy_formula_check

#-----------------------------------------------------------------------------
# unit tests

def test_parse1():
    sfc = SympyFormulaChecker()
    ans = 'X/2'
    r = sfc.parse_input(ans)
    print r
    assert(r==sympy.sympify("Matrix([[0,1],[1,0]])/2"))

def test_parse2():
    sfc = SympyFormulaChecker()
    ans = '[[0,1],[1,0]]/2'
    r = sfc.parse_input(ans)
    print r
    assert(r==sympy.sympify("Matrix([[0,1],[1,0]])/2"))

def test_make_samples():
    sfc = SympyFormulaChecker()
    r = sympy.sympify('exp(1-p)')
    s = sympy.sympify('cos(1-g)')
    samples = sfc.sympy_make_samples([r,s])
    assert(samples=='p,g@0,0:0.999,0.999#20')

def test_form_eq1():
    sfc = SympyFormulaChecker()
    r = sympy.sympify('exp(1-p)')
    s = sympy.sympify('cos(1-g)')
    eq = sfc.sympy_is_formula_equal(r, s)
    assert(eq==False)

def test_form_eq2():
    sfc = SympyFormulaChecker()
    r = sfc.parse_input('sqrt(1-sin(1-p)^2)')
    s = sympy.sympify('cos(1-p)')
    eq = sfc.sympy_is_formula_equal(r, s)
    assert(eq==True)

def test_sfc1():
    sfc = SympyFormulaChecker()
    ans = 'sqrt(1-sin(1-p)^2)'
    exp = 'cos(1-p)'
    ret = sfc.sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok']==True)

def test_sfc2():
    sfc = SympyFormulaChecker()
    ans = 'qux(1-sin(1-p)^2)'
    exp = 'cos(1-p)'
    ret = sfc.sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok']==False)

def test_sfc4():
    ans = "[[2/3,sqrt(1/3)],[sqrt(1/3),1/3]]"
    exp = "[[1,0],[0,0]]" 
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok']==False)

def test_sfc5():
    ans = "eye(2)"
    exp = "[[1,0],[0,1]]" 
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok']==True)

def test_sfc6():
    ans = "eye(3)"
    exp = "[[1,0],[0,1]]" 
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok']==False)

def test_sfc7():
    ans = "exp(2.0*pi*i)" 
    exp = "1.0"
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sfc3a():
    ans = 'foo(-sin(1-p)^2)'
    exp = 'None'
    ret = sympy_formula_check(exp, ans, options="check_none=1")
    print ret['msg']
    assert(ret['ok']==False)

def test_sfc4a():
    ans = 'none'
    exp = 'None'
    ret = sympy_formula_check(exp, ans, options="check_none=1")
    print ret['msg']
    assert(ret['ok']==True)

def test_sfc_list1():
    ret = sfc.sympy_evaluator({'x':2},{},"[x+1,x*4]")
    assert(ret==[3,8])

def test_sfc_list2():
    ret = sympy_formula_check(expect="[XX,YZ]", ans="[YZ,XX]")
    print ret['msg']
    assert(not ret['ok'])

def test_sfc_list3():
    ret = sympy_formula_check(expect="[XX,YZ]", ans="[XX, YZ+0]")
    print ret['msg']
    assert(ret['ok'])

def test_sfc_complex1():
    ret = sympy_formula_check('4+5*sqrt(-1)', "4+5*i")
    print ret['msg']
    assert(ret['ok'])

def test_sfc_complex2():
    expect="exp(i*(15-w)*pi/4)"
    ans = "exp(i*(15-w)*pi/3)"
    ret = sympy_formula_check(expect, ans)
    print ret['msg']
    assert(not ret['ok'])
    
#-----------------------------------------------------------------------------

def sympy_check_nocomm(expect, ans, options=None):
    '''
    Special version of sympy_formula_check for decomposing clifford ops problem, in which symbols should not commute
    '''
    ncs = 'AghIi'
    symtab = dict([[x,sympy.Symbol(x,commutative=False)] for x in ncs])
    mysfc = SympyFormulaChecker(extra_symbols=symtab)
    return mysfc.sympy_formula_check(expect, ans, options="only_symbolic=1")

def test_sympy_check_tol1():
    expect = "0.4*m"
    options="samples='m@1:10#50'!tolerance=0.05"
    ans = "0.399123*m"
    ret = sympy_formula_check(expect, ans, options=options)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_nocomm1():
    expect = "-h*g"
    ans = "-g*h"
    ret = sympy_check_nocomm(expect, ans)
    print ret['msg']
    assert(not ret['ok'])

def test_sympy_check_nocomm2():
    expect = "-h*g"
    ans = "-h*g"
    ret = sympy_check_nocomm(expect, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_nocomm3():
    expect = "[[I,g],[h,-h*g]]"
    ans = "[[I,g],[h,-g*h]]"
    ret = sympy_check_nocomm(expect, ans)
    print ret['msg']
    assert(not ret['ok'])

def test_sympy_check_nocomm4():
    expect = "[[I,g],[h,-h*g]]"
    ans = "[[I,g],[h,-h*g]]"
    ret = sympy_check_nocomm(expect, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_nocomm5():
    expect = "[[I,g],[h,-h*g]]"
    ans = "g*h"
    ret = sympy_check_nocomm(expect, ans)
    print ret['msg']
    assert(not ret['ok'])

#-----------------------------------------------------------------------------

def sympy_check_quantum(expect, ans, options=None):
    '''
    Special version of sympy_formula_check for quantum states.
    Allows ket notation (with numbers inside kets).
    '''
    sfc = SympyFormulaChecker(do_quantum=True)
    return sfc.sympy_formula_check(expect, ans, options)

def test_sympy_check_quantum1():
    expect="a*|00> + a*exp(i*theta)* |01>+b*|11> + b* exp(i*theta)*|10>"
    ans="a*|00> + a*exp(i*theta)* |01> + 1.0*b* exp(I*theta)*|10> + b*|11>"
    ret = sympy_check_quantum(expect, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_quantum2():
    expect="a *exp(i*theta)*|0> + b*|1>" 
    options="altanswer='a *|0> + b *exp(-i*theta)*|1>'"
    ans="a *|0> + b *exp(-i*theta)*|1>"
    ret = sympy_check_quantum(expect, ans, options)
    print ret['msg']
    assert(ret['ok'])

def test_sfc8a():
    ans = "exp(2.0*pi*i)" 
    exp = "1.0"
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_quantum3():
    expect="a*|00> + a*exp(i*theta)* |01>+b*|11> + b* exp(i*theta)*|10>"
    ans="a*|00> + a*exp(i*theta+2*pi*i)* |01>+b*|11> + b* exp(i*theta)*|10>"
    ret = sympy_check_quantum(expect, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sympy_check_quantum4():
    expect="a*|0>"
    ans="a*exp(2.0001*pi*i)* |0>"
    ret = sympy_check_quantum(expect, ans)
    print ret['msg']
    assert(ret['ok'])

def test_parse_input_2pii():
    sfc = SympyFormulaChecker(do_quantum=True)
    x = sfc.parse_input('a*e^(I*pi*2)*|0>')
    assert(str(x)=='a*|0>')

def test_parse_input_2pii2():
    x = sfc.parse_input("a*exp(2.0001*pi*i)")
    print "x=",x
    print type(x)
    y = x.evalf(subs={sympy.Symbol('a'):1})
    print "y=",y
    assert((abs(y)-1.0)<0.001)

def test_sfc8():
    ans = "exp(2.0*pi*i)" 
    exp = "1.0"
    ret = sympy_formula_check(exp, ans)
    print ret['msg']
    assert(ret['ok'])

def test_sfc9():
    expect="[[1,1],[1,-1]]/sqrt(2)" 
    options="altanswer='[[1,-1],[1,1]]/sqrt(2)'" 
    ans = "[[1,-1],[1,1]]*sqrt(2)/2"
    ret = sympy_formula_check(expect, ans, options)
    print ret['msg']
    assert(ret['ok'])
  
def test_sfc10():
    expect="[YZ,-XZ,IY]"
    ans="[YZ,XZ,IY]"
    ret = sympy_formula_check(expect, ans)
    print ret['msg']
    assert(not ret['ok'])
    
def test_sfc11():
    # make sure ZZ doesn't parse to become sympy.polys.domains.pythonintegerring.PythonIntegerRing, which breaks things
    expect="[ZI,IZ]"
    ans="[ZI,ZZ]"
    ret = sympy_formula_check(expect, ans)
    print ret['msg']
    assert(not ret['ok'])
    
def test_sfc12():
    # see what error is output if the matrix input has the wrong size
    expect="([[1,-1],[1,1]])/sqrt(2)" 
    ans="[[0, -i,0,0],[-i,0,0,0],[0,0,0,-i],[0,0,-i,0]]"
    ret = sympy_formula_check(expect, ans)
    print ret['msg']
    assert(not ret['ok'])
    assert("could not evaluate" not in ret['msg'])
