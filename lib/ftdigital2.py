#!/usr/bin/python
#
# File:   ftdigital2.py
# Date:   01-Mar-12
# Author: I. Chuang <ichuang@mit.edu>
#
# Simulate digital circuit; determine effect of faulty gates
# 09-Sep-2017 ftdigital2.py: version for Fall 2017 8.370, with reversible gates and circuits

import numpy
import os, sys, string, re
import sympy

from lxml import etree
from cStringIO import StringIO
from collections import OrderedDict

#-----------------------------------------------------------------------------

global SEQ
SEQ = [0]

global FAULTS
FAULTS = {}	# dict, with keys = input Signal or string name of gate ident(); val = error sympy Symbol

#-----------------------------------------------------------------------------
# math unicode to ascii

def MathUnicodeToAscii(ustr, debug=False):
    '''
    mathjax \tt format generates funny unicode; convert to ascii
    so that folks can cut and paste from solutions.
    '''
    try:
        asc = str(ustr)
        return asc
    except Exception as err:
        pass

    ns = []
    for ch in ustr:
        if ord(ch)<128:
            ns.append(ord(ch))
        elif ord(ch)>0x1D7F5:
            ns.append(ord(ch)-0x1D7F6+ord('0'))	# numbers
        elif ord(ch)>(0x1d68E-5):
            ns.append(ord(ch)-0x1D68E+ord('e'))	# lowercase
        elif ord(ch)>(0x1d678-9):
            ns.append(ord(ch)-0x1D678+ord('I'))	# caps
        else:
            ns.append(ord(ch)-0x2217+ord('*'))	# operators
    try:
        asc = ''.join(map(chr, ns))
        return asc
    except Exception as err:
        if debug:
            print "unicode translation failed"
            print "ns=", ns
            raise
        pass

    return ustr	# can't do anything

#-----------------------------------------------------------------------------
# signal class

class Signal(object):
    '''
    Abstract representation of a signal, which carries information between CircuitElement instances.
    Mainly a container for a list of objects which can be strings, numbers, or sympy Symbols.
    Iterable.
    '''
    def __init__(self,val):
        if type(val)==list:
            self.val = [Signal(x) for x in val]	# if argument is list of things, then make signals of those things
            self.__iter__ = self.val.__iter__
        elif type(val)==str:
            self.val = sympy.Symbol(val)
        else:
            self.val = val

    def ident(self):				# string name of signal
        # if type(self.val)==int: return '%d' % self.val
        # if type(self.val)==str: return self.val
        if type(self.val)==sympy.Symbol: return str(self.val)
        return self.val.ident()

    def make_id(self):
        if isinstance(self.val,sympy.Symbol): return self.val.name
        raise Exception,"No id for symbol %s (type=%s)" % (self,type(self.val))

    id = property(make_id,None,None,'unique id for this signal')

    def eval(self,args=None):			# numerical evaluation
        if type(self.val)==int: return self.val
        if type(self.val)==str and args and args.has_key(self.val): return args[self.val]
        if type(self.val)==sympy.Symbol: return self.val
        return self.val.eval(args)

    def dots(self,nodemap,expand=False):	# dummy routine for graph viz - recursion termination
        return ''

    def clear_dots(self):			# dummy routine for graph viz - recursion termination
        return ''

    def subs(self, val_dict):
        '''
        substitutions into sympy equations
        '''
        if type(self.val)==list:
            return [x.subs(val_dict) for x in self.val]
        return self.val.subs(val_dict)

    def make_eq(self,args=None):			# symbolic variable for signal
        if hasattr(self.val,'eq'):
            return self.val.eq
        if type(self.val)==sympy.Symbol: return self.val
        if type(self.val)==str: return sympy.Symbol(self.val)
        return self.val

    eq = property(make_eq,None,None,'symbolic variable for signal')	# sympy equation representation of signal

    def __eq__(self,other):
        if type(other)==type(self):
            return other.val == self.val
        if type(other)==sympy.Symbol:
            return other==self.val
        return False

    def __str__(self):
        return str(self.val)
    __repr__ = __str__

    def __len__(self):
        if type(self.val)==list: return len(self.val)
        return 1

    def __getitem__(self,k):
        if type(self.val)==list:
            return self.val[k]
        if k==0:
            return self.val
        raise Exception,'[Signal] not list: cannot return item %s of signal %s' % (k, self)

    #def __iter__(self):
    #    return self.val.__iter__()

    def next(self):
        return self.val.net()

#-----------------------------------------------------------------------------

class CircuitElement(object):
    output_length = 1

    def __init__(self,*args,**kwargs):
        '''
        Generate circuit element.  Input arguments should be the inputs to the gate.
        An optional keyword argument "outputs" can give name(s) to the output(s), which may be used
        when drawing the graph for the circuit.

        A CircuitElement may have multiple outputs, in which case its output is a Signal of a list.
        The len of a CircuitElement is its number of outputs.
        '''
        self.make_id()
        #self.args = [(x if isinstance(x,CircuitElement) else Signal(x)) for x in args]
        atypes = [str, list, sympy.Symbol]
        self.args = [(Signal(x) if (type(x) in atypes) else x) for x in args]

        if 'outputs' in kwargs:			# if output argument names are specified 
            outarg = kwargs['outputs']
            if not type(outarg)==list: outarg = [outarg]
            self.outputs = [(Signal(x) if (type(x) in atypes) else x) for x in outarg]

        if hasattr(self,'make_circ'):
            self.make_circ()
        self.dots_done = False	# flag to help make sure graph doesn't have duplicate node paths

    def __len__(self):
        return self.output_length

    def make_id(self):
        global SEQ
        self.id = '%s%d' % (self.nodename,SEQ[0])
        SEQ[0] += 1
        
    def subs(self, *args):
        return self.eq.subs(*args)

    #def make_eq(self):
    #    return None
    #eq = property(make_eq,None,None,'return sympy equation or symbol')

    def eval(self,*argvals):	
        '''
        Evaluate equation for circuit.  Arguments should be a list of inputs for the circuit, with
        the list length matching the expected number of arguments.  Alternatively, a single argument
        may be provided, which is a dict for the sympy variables used in the equation for the circuit.
        '''
        if not argvals: return self.eq
        elif type(argvals[0])==dict:
            avmap = argvals[0]
        elif not len(argvals)==len(self.args):
            raise Exception,"[CircuitElement] error: argument values %s don't match args %s" % (argvals,self.args)
        else:
            avmap = dict(zip([x.eq for x in self.args],argvals))
        # print "avmap = ",avmap
        return self.eq.subs(avmap)

    def __getitem__(self,k):
        return self.args[k]

    def ident(self):
        return "[%s](%s)" % (self.id,','.join([k.ident() for k in self.args]))

    def clear_dots(self):
        self.dots_done = False
        for a in self.args: a.clear_dots()

    def dots(self,nodemap={},expand=False):
        '''
        Return a GraphViz "dot" language description of our node.
        If expand=True, then expand circuit (into base components - those without circ attrib) for graph
        '''
        dots = ''
        if self.dots_done: return dots		# don't generate node path twice
        self.dots_done = True

        if expand and hasattr(self,'circ'):
            dots += '\t\t// %s\n' % self
            dots += self.circ.dots(nodemap,expand)
            return dots

        for a in self.args:
            # dots += '// %s\n' % self
            dots += '%s -> %s;	// %s\n' % (nodemap[a] if a in nodemap else a.id,self.id,self)
        for a in self.args:
            dots += a.dots(nodemap,expand)
        return dots

    def make_eq(self):
        '''
        Return sympy equation for the output.  If the gate is marked as being faulty, then
        return a unique error variable instead of the normal output.
        '''
        if not hasattr(self,'make_eq_nofault'):
            raise Exception,'[CircuitElement]: missing make_eq_nofault definition for %s' % self

        global FAULTS
        # print "Evaluating nand%d(%s,%s)" % (self.id,self.args[0],self.args[1])

        if self.id in FAULTS:				# if fault, then output error symbol instead
            errsym = sympy.Symbol('e%s' % self.id)
            FAULTS[self.id] = errsym			# also record the error symbol used
            return errsym

        return self.make_eq_nofault()

    eq = property(make_eq,None,None,'return sympy equation or symbol for the circuit element')

    def __str__(self):
        return '%s(%s)' % (self.gatename,','.join([str(x) for x in self.args]))
    __repr__ = __str__


#-----------------------------------------------------------------------------

class nand(CircuitElement):
    '''
    NAND gate element.

    Examples:

       n = nand('x','y')
       n.eval(x=0,y=1)

       n = nand(0,1)
       n.eval()

    '''
    gatename = 'Nand'
    nodename = 'Nand'	# for the graph, and for the CircuitElement id

    def make_eq_nofault(self):
        (x,y) = self.args
        return ~(x.eq & y.eq)	# return sympy equation
        

class andgate(CircuitElement):
    '''
    AND gate element.
    '''
    gatename = 'And'
    nodename = 'And'	# for the graph, and for the CircuitElement id

    def make_eq_nofault(self):
        (x,y) = self.args
        return (x.eq & y.eq)	# return sympy equation

#-----------------------------------------------------------------------------

class orgate(CircuitElement):
    '''
    OR gate element.
    '''
    gatename = 'Or'
    nodename = 'Or'	# for the graph, and for the CircuitElement id

    def make_eq_nofault(self):
        (x,y) = self.args
        return (x.eq | y.eq)	# return sympy equation

class notgate(CircuitElement):
    '''
    NOT gate element.
    '''
    gatename = 'Not'
    nodename = 'Not'	# for the graph, and for the CircuitElement id

    def make_eq_nofault(self):
        return ~self.args[0].eq		# return sympy equation

#-----------------------------------------------------------------------------
# reversible gates (beyond not)

class xorgate(CircuitElement):
    '''
    XOR gate element.
    '''
    gatename = 'xor'
    nodename = 'xor'	# for the graph, and for the CircuitElement id

    def make_eq_nofault(self):
        (x,y) = self.args
        return (( x.eq & (~y.eq)) | ( (~x.eq) & y.eq ))	# return sympy equation

class swapgate(CircuitElement):
    '''
    swap gate element.
    '''
    gatename = 'swap'
    nodename = 'swap'	# for the graph, and for the CircuitElement id
    output_length = 2

    def make_eq_nofault(self):
        (x,y) = self.args
        return Signal([y, x])	# tricky - return signal list instead of just a signal

class cnot(CircuitElement):
    '''
    swap gate element.
    '''
    gatename = 'cnot'
    nodename = 'cnot'	# for the graph, and for the CircuitElement id
    output_length = 2

    def make_eq_nofault(self):
        (x,y) = self.args
        # return Signal([x, (~x.eq) & y.eq | (x.eq & ~y.eq)])
        return Signal([x, xorgate(x, y)])

class toffoli(CircuitElement):
    '''
    toffoli gate element.
    '''
    gatename = 'toffoli'
    nodename = 'toffoli'	# for the graph, and for the CircuitElement id
    output_length = 3

    def make_eq_nofault(self):
        (x,y,z) = self.args
        return Signal([x, y, xorgate(Signal(x.eq & y.eq), z) ])	# return signal list

class fredkin(CircuitElement):
    '''
    fredkin gate element.
    '''
    gatename = 'fredkin'
    nodename = 'fredkin'	# for the graph, and for the CircuitElement id
    output_length = 3

    def make_eq_nofault(self):
        (x,y,z) = self.args
        return Signal([x, x.eq & z.eq | (~x.eq) & y.eq, x.eq & y.eq | (~x.eq) & z.eq ])	# return signal list

#-----------------------------------------------------------------------------

class demux_two(CircuitElement):
    '''
    2-input demultiplexer (for a student exercise)
    '''
    gatename = 'demux2'
    nodename = 'demux2'
    output_length = 3

    def make_eq_nofault(self):
        (a, b, _, _) = self.args	# assume c=0, d=0
        return Signal([~a.eq & ~b.eq, a.eq & ~b.eq, ~a.eq & b.eq, a.eq & b.eq])

#-----------------------------------------------------------------------------
# maj gate

class maj(CircuitElement):
    '''
    Majority voting gate.  Three inputs, one output.
    '''

    nodename = 'm'

    def make_circ(self):
        (x,y,z) = self.args
        q = nand(nand(x,y),nand(y,z))
        self.circ = nand(nand(q,q),nand(x,z))

    def make_eq(self):
        return self.circ.eq

    eq = property(make_eq,None,None,'return sympy equation or symbol')
    def __str__(self):
        return 'maj(%s,%s,%s)' % (self.args[0],self.args[1],self.args[2])
    __repr__ = __str__
    def ident(self):
        return self.circ.ident()

#-----------------------------------------------------------------------------
# dicts of gate function sets

all_functions = {'and': andgate,
                 'or': orgate,
                 'not': notgate,
                 'nand': nand,
                 'xor': xorgate,
                 'maj': maj,
                 'swap': swapgate,
                 'cnot': cnot,
                 'fredkin': fredkin,
                 'toffoli': toffoli,
}

reversible_functions = {'not': notgate,
                        'swap': swapgate,
                        'cnot': cnot,
                        'fredkin': fredkin,
                        'toffoli': toffoli,
}

#-----------------------------------------------------------------------------
# level 1 gates

class MultipleCircuitElement(CircuitElement):
    '''
    Multiple a CircuitElement: each signal -> 3 signals.
    n inputs -> 3xn inputs,
    1 output -> 3x1 output
    '''
    def make_circ(self):
        '''
        One circuit for each output.
        '''
        argsets = zip(*self.args)			# TODO: allow permutations
        self.circ = [self.gate(*x) for x in argsets]	# one 1-output gate for each output

    def __str__(self):
        return str(self.circ)
    __repr__ = __str__

    def make_eq(self):
        '''
        Return list of sympy equations for the outputs
        '''
        return [x.eq for x in self.circ]
    eq = property(make_eq,None,None,'return sympy equation or symbol')

    def eval(self,*argvals):
        '''
        Evaluate equations for circuit.  The arguments should be lists of gate arguments.
        Alternatively, it can be a (single) dict, specifying values for sympy symbols.
        '''
        varset = sympy.flatten(self.args)	# list of variables which are input arguments (sympy.Symbol)
        varset = [(v.val if isinstance(v,Signal) else v) for v in varset]	# turn Signal instances back into sympy Symbols

        if not argvals: argvsets = {}
        elif type(argvals[0])==dict:
            argvsets = argvals[0]
        elif not len(argvals)==len(self.args):
            raise Exception,"[MultipleCircuitElement] error: argument values %s don't match args %s" % (argvals,self.args)
        else:
            valset = sympy.flatten(argvals)		# list of values to be assigned to input argument variables
            argvsets = dict(zip(varset,valset))		# dict matching all variables and values

        # if fault marked for any input then handle that using argvsets
        global FAULTS
        # faultset = [ (x.val if type(x)==Signal else x) for x in FAULTS]
        for v in varset:
            if v in FAULTS:
                errsym = sympy.Symbol('%sErr' % v)
                FAULTS[v] = errsym
                argvsets[v] = errsym

        #print "[MultipleCircuitElement] argvsets = ",argvsets
        if not argvsets: return self.eq
        return [c.eval(argvsets) for c in self.circ]

    def ident(self):
        return [x.ident() for x in self.circ]

    #def plot2svg(self,url_root,imdir):
    #    # generate a canonical name for this plot, based on the adjacency matrix entries
    #    #fn = "ft-circuit-%s.png" % hash(''.join(self.ident()))
    #    #print "fn = ",fn
    #    #pfn = '%s/%s' % (imdir,fn)
    #    #url = '/%s/%s' % (url_root,fn)
    #    ##if os.path.exists(pfn):
    #    ##    return url
    #    svg = self.plot()
    #    return svg

    def plot(self,expand=False,fn=None,**kwargs):
        '''
        Generate a plot of the circuit using GraphViz and dot
        If filename (fn) is given, then save the plot there.

        The inputs and outputs have to be handled specially, since they are blocks.

        digraph circ
        {
        xin [label="<a>a|<b>b|<c>c" shape=record];
        yin [label="<d>d|<e>e|<f>f" shape=record];
        "xin":a -> n1;
        "yin":d -> n1;
        "xin":b -> n2;
        "yin":e -> n2;
        "xin":c -> n3;
        "yin":f -> n3;
        }

        '''

        # make a map of symbol name to graph node name for the inputs and outputs
        nodemap = {}
        for arg in self.args:
            argno = self.args.index(arg)+1
            nodemap.update(dict([ (x,'"in%d":%s' % (argno,x.id)) for x in arg ]))
        #print "nodemap = ",nodemap
        self.nodemap = nodemap

        dots = 'digraph circ\n{\n'
        for arg in self.args:		# nodes for input blocks
            argno = self.args.index(arg)+1
            dots += 'in%d [label="%s" shape=record];\n' % (argno,'|'.join([ '<%s>%s' % (x.id,x.id) for x in arg]))

        # node for output block
        if (hasattr(self,'outputs') and type(len(self.outputs[0])==len(self.circ))): outputs = self.outputs[0]
        else: outputs = range(1,len(self.circ)+1)
        dots += 'out [label="%s" shape=record];\n' % ('|'.join(['<%s>%s' % (x,x) for x in outputs]))

        # nodes for circuits going to outputs
        for c in self.circ:
            cno = outputs[self.circ.index(c)]
            dots += '%s -> "out":%s\t// %s\n' % (c.circ.id if expand and hasattr(c,'circ') else c.id,cno,c)

        # nodes for each circuit
        for c in self.circ:
            c.clear_dots()			# clear record of dots; makes sure only generated once for each node
            dots += c.dots(nodemap,expand)

        dots += '}\n'

        if 0:
            fp = open('circ.dot','w')
            fp.write(dots)
            fp.close()
        
        sfp = StringIO()
        g = gviz.AGraph(string=dots)
        g.draw(path=sfp, format='svg', prog='dot')
        svg = sfp.getvalue()
        svg = svg.replace('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n','')
        svg = svg.replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">','')
        # svg = svg.replace('<svg ', '<svg  xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ')
        svgxml = etree.fromstring(svg)

        # for some reason, the edX platform won't pass through nested <g>'s in the SVG.  That ruins the output
        # of graphviz's SVG.  Let's work around that by flattening the <g>'s, putting the translate transformation
        # into each of the main g's children.
        topg = svgxml.find('.//{http://www.w3.org/2000/svg}g[@class="graph"]')
        xfm = topg.get('transform')
        for childg in topg.findall('.//{http://www.w3.org/2000/svg}g'):
            childg.set('transform', xfm)
            topg.addnext(childg)
        #whitepoly = topg.find('.//{http://www.w3.org/2000/svg}polygon')
        #topg.remove(whitepoly)

        return etree.tostring(svgxml, pretty_print=True, method="html")

        #if fn:
        #    g = gviz.AGraph(string=dots)
        #    g.draw(path=fn,prog='dot')
        #
        #return dots
            
    def is_eq_faulty_old(self,beq):
        '''
        A beq is a boolean equation for a signal in a block.

        A beq is bad if it depends on an error.  Ideally, this could be determined by
        simplifying the boolean formula and finding that it does not depend on the errors.
        However, the simplifier is flaky, and does not properly simplify instances like
        Or(And(x,e),x) to become x, as it should.
        
        Thus, we have to check for dependence by substituting in T/F values for each of the
        errors and making sure the beq does not change.

        Note that a faulty gate might not get evaluated, it it appears before another gate which is faulty.
        '''
        global FAULTS
        #errors = [sympy.Symbol('e%s' % x) for x in FAULTS]
        errors = []				# symbols for the faults
        for v in FAULTS.values():
            if v: errors.append(v)

        # outset = [ y.name for y in list(beq.atoms()) ]
        # for e in errors:
        #     if e in outset: return 1
        
        def bin2list(x,nbits): return [ (b=='1') for b in (bin(x)[2:].rjust(nbits, '0')) ]

        ev = None
        nbits = len(FAULTS)
        for k in range(2**nbits):
            newev = beq.subs(dict(zip(errors,bin2list(k,nbits))))
            if not ev: ev = newev
            else: 
                if not (ev==newev): 
                    return 1

        return 0
            
    def is_eq_faulty(self, beq, the_faults=None, varset=None):
        '''
        A beq is a boolean equation for a signal in a block.

        A beq is bad if it depends on an error.  Ideally, this could be determined by
        simplifying the boolean formula and finding that it does not depend on the errors.
        However, the simplifier is flaky, and does not properly simplify instances like
        Or(And(x,e),x) to become x, as it should.
        
        Thus, we have to check for dependence by substituting in T/F values for each of the
        errors and making sure the beq does not change.

        Note that a faulty gate might not get evaluated, if it appears before another gate which is faulty.
        
        beq = sympy boolean equation
        the_faults = (optional) dict of symbol: value listing error variables
        varset = (optional) dict of symbol: value listing non-error variables to substitute before comparison
        '''
        if the_faults is None:
            global FAULTS
            the_faults = FAULTS

        #errors = [sympy.Symbol('e%s' % x) for x in FAULTS]

        errors = []				# symbols for the faults

        for v in the_faults.values():
            if v is not None: errors.append(v)

        # outset = [ y.name for y in list(beq.atoms()) ]
        # for e in errors:
        #     if e in outset: return 1
        
        def bin2list(x,nbits): return [ (b=='1') for b in (bin(x)[2:].rjust(nbits, '0')) ]

        ev = None
        nbits = len(the_faults)
        for k in range(2**nbits):
            vars = dict(zip(errors,bin2list(k,nbits)))
            if varset is not None:
                vars.update(varset)
            newev = beq.subs(vars)
            if ev is None: ev = newev
            else: 
                if not (ev==newev): 
                    return 1

        return 0

    is_eq_faulty2 = is_eq_faulty

    def gate_id_set(self):
        gateidset = list(set(re.findall('\[([^]]+)\]',''.join(self.ident()))))
        gateidset.sort()
        # print "gateidset = ",gateidset
        return gateidset

    def ft_check1(self,verbose=False):
        '''
        Determine single gate fault paths (ie single gate failures which lead to the output block having an error).
        Returns set of fault paths as a dict, with the key being the id of the gate which failed, and the
        values being the output with the error.
        '''
        global FAULTS
        gateidset = self.gate_id_set()
        
        faultset = {}
        for fault in gateidset:
            FAULTS = {fault:None}
            outblocks = self.eval()
            blocksbad = [self.is_eq_faulty(x) for x in outblocks]
            isbad = sum(blocksbad)>1
            if isbad:
                faultset[fault] = outblocks
        if verbose: print "%d fault paths" % len(faultset)
        return faultset

    def ft_check2(self,verbose=False):
        '''
        Determine double gate fault paths (ie two-gate failures which lead to the output block having an error).
        Returns set of fault paths as a dict, with the key being (id1,id2) of the gates which failed, and the
        values being the output with the error.
        '''
        global FAULTS
        gateidset = self.gate_id_set()
        locset = [(x.val if type(x)==Signal else x) for x in sympy.flatten(self.args)]
        locset += gateidset	# allow input arguments to also be faulty
        
        faultset = {}
        for j in range(len(locset)):
            fault1 = locset[j]
            for k in range(j):
                fault2 = locset[k]
                FAULTS = {fault1:None, fault2:None}
                outblocks = self.eval()
                blocksbad = [self.is_eq_faulty(x) for x in outblocks]
                isbad = sum(blocksbad)>1
                if isbad:
                    faultset[(fault1,fault2)] = outblocks
        if verbose: print "%d fault paths" % len(faultset)
        return faultset

    def ft_check3(self, verbose=False, varsets=None, do_nand=False):
        '''
        Determine double gate fault paths (ie two-gate failures which lead to the output block having an error).
        Returns set of fault paths as a dict, with the key being (id1,id2) of the gates which failed, and the
        values being the output with the error.
        
        varsets = logical values of variables to substitute in, as list of dicts of {symbol: truth_value}

        If varsets is given, then the output is checked for all varset dicts given.
        '''
        if do_nand:
            # check circuit assuming inputs are error-free unless explicitly marked with an error

            def invar(x, vn):
                return { sympy.Symbol(s): x for s in vn }

            def invar2(x, y):
                vabc = invar(x, 'abc')
                vdef = invar(y, 'def')
                vabc.update(vdef)
                return vabc
    
            # input block values
            values = [(True, True), (True, False), (False, True), (False, False)]
            varsets = [invar2(*v) for v in values]

        gateidset = self.gate_id_set()
        locset = [(x.val if type(x)==Signal else x) for x in sympy.flatten(self.args)]
        locset += gateidset	# allow input arguments to also be faulty
        
        faultset = {}
        for j in range(len(locset)):
            fault1 = locset[j]
            for k in range(j):
                fault2 = locset[k]
                (isbad, outblocks) = self.ft_check_one([ fault1, fault2 ], varsets)
                if isbad:
                    faultset[(fault1,fault2)] = outblocks
        if verbose: print "%d fault paths" % len(faultset)
        return faultset

    def ft_check_one(self, fault_list, varsets=None):
        global FAULTS

        FAULTS = { fault:None for fault in fault_list }
        outblocks = self.eval()		# generates boolean equation with faults

        if varsets is None:
            varsets = [None]

        for vs in varsets:
            blocksbad = [self.is_eq_faulty(x, varset=vs) for x in outblocks]
            isbad = sum(blocksbad)>1
            if isbad:
                return (isbad, outblocks)	# bad if bad for any of the varsets
        return (isbad, outblocks)
        

class tnand(MultipleCircuitElement):
    gate = nand
    nodename = 'tripple_nand'
        
class ftnand(MultipleCircuitElement):
    '''
    Fault-tolerant NAND gate, with three tripple NANDs, and three MAJ gates.
    '''
    nodename = 'ftnand'

    def make_circ(self):
        # 2x3 inputs which are duplicated for the three tripple NAND gates
        self.nandset = [tnand(*self.args) for x in range(3)]		# 9 output wires; route three each to a MAJ gate
        # output given by three MAJ gates
        self.circ = [ maj(*x) for x in zip(*[x.circ for x in self.nandset]) ]
        
#-----------------------------------------------------------------------------
# simple reversible circuit book keeping class
        
class CircuitGate(object):
    def __init__(self, funcname, args, argnames=None):
        self.funcname = funcname
        self.args = args
        self.argnames = argnames

    def plot_gate(self, plotter, xpos):
        a2y = plotter.circuit.argname2index
        yp = map(a2y, self.argnames)
        if self.funcname=="cnot":
            plotter.control_point(xpos, yp[0])
            plotter.not_point(xpos, yp[1])
            plotter.control_line(xpos, yp[0], yp[1])

        elif self.funcname=="swap":
            plotter.swap_point(xpos, yp[0])
            plotter.swap_point(xpos, yp[1])
            plotter.control_line(xpos, min(yp), max(yp))

        elif self.funcname=="toffoli":
            plotter.control_point(xpos, yp[0])
            plotter.control_point(xpos, yp[1])
            plotter.not_point(xpos, yp[2])
            plotter.control_line(xpos, min(yp), max(yp))

        elif self.funcname=="fredkin":
            plotter.control_point(xpos, yp[0])
            plotter.swap_point(xpos, yp[1])
            plotter.swap_point(xpos, yp[2])
            plotter.control_line(xpos, min(yp), max(yp))

        elif self.funcname=="not":
            plotter.not_point(xpos, yp[0])

    def __unicode__(self):
        return "%s(%s)" % (self.funcname, ','.join(self.argnames))
    __repr__ = __unicode__
    __str__ = __unicode__

class CircuitKeeper(object):
    def __init__(self, inputs):
        '''
        inputs = ordered list of input names
        '''
        self.inputs = inputs
        self.nbits = len(inputs)
        self.gates = []

    def add_gate(self, gate):
        self.gates.append(gate)

    def argname2index(self, argname):
        return len(self.inputs) - self.inputs.index(argname) - 1

    def plot(self, html=False, labels=None, **params):
        '''
        Generate plot image of circuit, using CircuitPlot from sympy.physics.quantum
        return SVG / HTML
        '''
        labels = labels or self.inputs
        cp = CircuitPlotSVG(self, input_names=labels, **params)
        if html:
            return '<center>%s</center>' % str(cp.svg)
        else:
            return str(cp.svg)

    def __unicode__(self):
        return unicode(self.gates)
    
    __repr__ = __unicode__
    __str__ = __unicode__

#-----------------------------------------------------------------------------
# simple functional evaluator

class FuncEval(object):
    '''
    Evaluate a functional composition string like not(and(not(x), not(y)))
    allowing a pre-specified list of functions, and returning
    the function's sympy output.

    Allow allow the expression to be a list, e.g. [True, not(False)]

    True and False (and true and false) are accepted literals.
    '''
    def __init__(self, functions, verbose=False):
        '''
        functions = dict of allowed functions, with name: function
        '''
        self.allowed_functions = functions
        self.separators = [",", "(", ")", "[", "]"]
        self.literals = {"true": True,
                         "false": False,
                         }
        self.verbose = verbose
        self.scalars_evaluated = []
        self.circuit = CircuitKeeper([])
        if self.verbose:
            print "[FunvEval] allowed functions = %s" % self.allowed_functions
        
    def add_function(self, funcname, newfunc):
        self.allowed_functions[funcname] = newfunc

    def eval_composition(self, expr, input_signals, level=0):
        '''
        Evaluate a multi-line sequence of expressions as a composition upon itself.
        Inputs to the second expression are the outputs of the first, etc.
        Useful for a cascade of reversible gates acting on a fixed number of bits.

        input_signals = list of string names of input signals, or 
                        ordered dict of the input signals, with name: signal
        '''
        local_env = {}
        if isinstance(input_signals, str) or isinstance(input_signals, list):
            input_signals = OrderedDict([(x,  Signal(x)) for x in input_signals])
        local_env.update(input_signals)
        nline = 0
        self.circuit = CircuitKeeper(input_signals.keys())
        self.intermediate_outputs = [local_env.copy()]
        for exprline in expr.split('\n'):
            exprline = exprline.strip()
            if not len(exprline):		# skip blank lines
                continue
            nline += 1
            self.scalars_evaluated = []			# holds names of signals input to gate
            fout = self.eval(exprline, env=local_env)
            # assume that the bits acted upon (ie scalars_evaluated) are the bits output
            nin = len(self.scalars_evaluated)
            ndistinct = len(set(self.scalars_evaluated))
            if not ndistinct==nin:
                raise Exception("[FuncEval] Error!  Cannot have gate with wire going to two inputs: #in=%s, #distinct=%s" % (nin, ndistinct))
            if not len(fout)==len(self.scalars_evaluated):
                raise Exception("[FuncEval] Error!  Cannot eval_composition with #out=%s, but #in=%s, in expr=%s" % (len(fout),
                                                                                                                     len(self.scalars_evaluated),
                                                                                                                     exprline))
            # if scalars_evaluated = [b,a] for [fout0, fout1] and inputs=[a,b]
            # then assign a'=fout1, b'=fout0

            # scalars_evaluated might be [a,d,c] for [fout0, fout1, fout2], and inputs=[a,b,c,d]
            # in that case, we want to assign a'=fout0 b'=b, c'=fout2, d'=fout1

            if isinstance(fout.eq, Signal):
                output_list = fout.eq
            else:
                output_list= [Signal(fout.eq)]
            for scalar, oneout in zip(self.scalars_evaluated, output_list):
                local_env[scalar] = oneout
                if self.verbose:
                    print "[FuncEval.eval_composition] line %s setting %s = %s" % (nline, scalar, oneout)
                    # sys.stdout.flush()
            self.intermediate_outputs.append(local_env.copy())

        self.result = Signal([local_env[x] for x in input_signals])

        return local_env

    def eval(self, expr, env=None, level=0):
        '''
        Main method for evaluating an expression.  Sample expressions:

        and(x,y)
        not(and(not(x), not(y)))
        [ True, and(x,y), not(z)]
        nand(nand(nand(nand(x,y),nand(y,z)),nand(nand(x,y),nand(y,z))),nand(x,z))
        swap(x,y)
        [ q = swap(x,y); q.0 ]
        [ q = swap(x,y); [ q.0, q.1 ] ]

        '''
        self.env = env or {}		# evaluation environment - for defined literals
        if self.verbose:
            print "[FunvEval] " + "--" * level + " evaluating expr = %s" % expr

        expr = expr.strip()
            
        if (level==0):
            self.original_expr = expr

        if expr.startswith('[') and expr.endswith(']') and ';' in expr:	# this is a multiline expression
            # allow expressions like [ q=swap(x,y); q.0 ]
            # the final return value is that of the last expression
            exprlist = expr[1:-1].split(';')
        elif '\n' in expr:
            exprlist = exprlist.split('\n')
        else:
            exprlist = [expr]

        if len(exprlist) > 1:
            self.exprlist = exprlist
            for sub_expr in exprlist:
                if not sub_expr:
                    continue
                ret = self.eval(sub_expr, env=self.env, level=level+1)
            return ret

        self.expr = expr

        # break expr into lhs and rhs if there is an equal sign
        if '=' in expr:
            (lhs, rhs) = expr.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()
        else:
            lhs = None
            rhs = expr

        # evaluate rhs
        tokens = self.tokenize(rhs)
        self.tokens = tokens
        self.tk = 0
        if self.verbose:
            print "[FunvEval] tokens = %s" % self.tokens
        result = self.descend()
        self.result = result

        # assign lhs, if specified  (assumes lhs is just an alphanumeric string)
        if lhs:
            self.env[lhs] = result
            if self.verbose:
                print "[FunvEval] assigned lhs = %s" % result

        if self.verbose:
            if type(result)==list:
                print "[FuncEval.eval] result=%s" % result
            else:
                print "[FuncEval.eval] result=%s,  eq=%s" % (result, result.eq)
        return result

    @property
    def curtok(self):
        return self.tokens[self.tk]
    
    @property
    def nexttok(self):
        if self.tk+1 < len(self.tokens):
            return self.tokens[self.tk+1]
        return None

    def next(self):
        if self.tk+1 >= len(self.tokens):
            raise Exception("[FuncEval] Premature of expression, %s" % self.expr)
        self.tk += 1

    def eval_function(self, funcname, level):
        '''
        Continue evaluating tokens to construct a function evaluation
        '''
        self.next()
        if not funcname in self.allowed_functions:
            raise Exception("[FuncEval] unknown function %s in expression %s" % (funcname, self.expr))
        if not self.curtok=="(":	# opening parenthesis
            raise Exception("[FuncEval] syntax error, expecting '(' after %s in %s" % (funcname, self.expr))
        self.next()
        args = []
        while not self.curtok == ")":	# process until closing parenthesis
            args.append(self.descend(level=level+1))
            self.next()
            if self.curtok == ",":
                self.next()
                continue
            elif self.curtok==")":
                break
        if self.verbose:
            print "[FuncEval] Applying %s to arguments %s" % (funcname, args)
        self.circuit.add_gate(CircuitGate(funcname, args, self.scalars_evaluated))
        return self.allowed_functions[funcname](*args)

    def eval_scalar(self, atok):
        '''
        Evaluate a scalar, return result.  This may be "x" or "a.2", which selects list element 2 of list a
        If the scalar is not known in the environment, then construct and return a sympy Symbol.
        '''
        self.scalars_evaluated.append(atok)
        if atok in self.env:
            return self.env[atok]
        m = re.match("([A-z]+)\.([0-9]+)", atok)		# syntax: a.2 selects list element 2 of a
        if m:
            sig = m.group(1)
            eidx = int(m.group(2))
            if sig in self.env and len(self.env[sig]) > eidx:
                retelem = self.env[sig]
                if isinstance(retelem, CircuitElement):
                    return retelem.eq[eidx]
                return retelem[eidx]
        if self.verbose:
            print "[FuncEval] eval_scalar: token=%s is a new symbol" % atok
        return sympy.Symbol(atok)

    def eval_list(self, level):
        '''
        Continue parsing tokens, and return a list of results.
        Leave token pointer pointing to next token after end of list.
        '''
        self.next()
        args = []
        while not self.curtok == "]":	# make list
            args.append(self.descend(level=level+1))
            self.next()
            if self.curtok == ",":
                self.next()
                continue
            elif self.curtok=="]":
                if self.nexttok is not None:
                    self.next()
                break
        return args		# return list

    def descend(self, level=0):
        if self.verbose:
            print "[FuncEval] descend " + "--" * level + "  curtok=%s, next=%s" % (self.curtok, self.nexttok)
        if self.curtok in self.literals:
            return self.literals[self.curtok]
        elif self.curtok == "[":
            return self.eval_list(level)
        elif self.curtok not in self.separators:
            if self.nexttok in [",", ")", "]", None]:
                return self.eval_scalar(self.curtok)
            return self.eval_function(self.curtok, level)
        else:
            raise Exception("[FuncEval] unexpected separator %s in expression %s" % (self.curtok, self.expr))

    def tokenize(self, expr):
        '''
        Return tokenized list of expr.  Parentheses and strings are separators.
        '''
        expr = expr.replace(' ', '')
        k = 0
        tokens = []
        a_token = ''
        while (k < len(expr)):
            c = expr[k]
            if c in self.separators:
                if a_token:
                    if a_token.lower() in self.literals:	# force literals to be lowercase
                        a_token = a_token.lower()
                    tokens.append(a_token)
                tokens.append(c)
                a_token = ''
            else:
                a_token += c
            k+= 1
        if a_token:
            tokens.append(a_token)
        return tokens
        
def test_funceval1():
    functions = all_functions
    fe = FuncEval(functions, verbose=True)
    z = fe.eval("and(x,y)")
    return fe, z
        
def test_funceval2():
    functions = all_functions
    fe = FuncEval(functions, verbose=True)
    z = fe.eval("not(and(not(x), not(y)))")
    print z.eq
    assert z.eq.subs({'x':True, 'y': False})==True
    assert z.eq.subs({'x':False, 'y': False})==False
    assert z.eq.subs({'x':True, 'y': True})==True
    assert z.eq.subs({'x':False, 'y': True})==True
    return fe, z

def test_funceval3():
    functions = all_functions
    fe = FuncEval(functions, verbose=True)
    z = fe.eval("[ True, and(x,y), not(z)]")
    print z
    assert len(z)==3
    return fe, z

def test_funceval4():
    functions = all_functions
    fe = FuncEval(functions, verbose=True)
    z = fe.eval("nand(nand(nand(nand(x,y),nand(y,z)),nand(nand(x,y),nand(y,z))),nand(x,z))")
    print z
    assert z.subs({'x':True, 'y':True, 'z':False})==True
    assert z.subs({'x':True, 'y':False, 'z':False})==False
    return fe, z

def test_funceval5():
    functions = all_functions
    fe = FuncEval(functions, verbose=True)
    z = fe.eval("swap(x,y)")
    print z
    assert str(z)=='swap(x,y)'
    return fe, z

def test_funceval6():
    fe = FuncEval(all_functions, verbose=True)
    z = fe.eval("[ q = swap(x,y); q.0 ]")
    print z
    assert str(z)=="y"
    return fe, z

def test_funceval7():
    fe = FuncEval(all_functions, verbose=True)
    z = fe.eval("[ q = swap(x,y); [ q.0, q.1 ] ]")
    print z
    assert str(z)=='[y, x]'
    return fe, z

def test_funceval8():
    fe = FuncEval(all_functions, verbose=True)
    inputs = OrderedDict([(x,  Signal(x)) for x in "ab"])
    z = fe.eval_composition("""
		cnot(a,b)
		cnot(b,a)
		cnot(a,b)
    """, inputs)
    print z
    assert fe.result.subs({'a':True, 'b':False})==[False, True]
    return fe, z

def test_funceval9():
    # fredkin made from toffolis
    fe = FuncEval(all_functions, verbose=True)
    inputs = "abc"
    z = fe.eval_composition("""
		toffoli(a,b,c)
		toffoli(a,c,b)
		toffoli(a,b,c)
    """, inputs)
    print z
    assert fe.result.subs({'a':False, 'b':False, 'c':True})==[False, False, True]
    assert fe.result.subs({'a':True,  'b':False, 'c':True})==[True,  True,  False]
    return fe, z

#-----------------------------------------------------------------------------
# check for boolean circuit equvalence

def check_equivalence(eqa, eqb, inputs, fixed_inputs=None, verbose=False, just_truth=False):
    '''
    Check truth tables of sympy equations eqa and eqb to see if they are identical, 
    on the list of inputs specified.

    inputs = list of names of inputs
    fixed_inputs = dict specifying name:val for inputs not to be varied
    just_truth = True if only a truth table for eqa should be returned (no comparison with eqb)
    '''
    table = {}
    assignments = {}
    tfbits = {True: "1", False: "0"}
    fixed_inputs = fixed_inputs or []
    def recursive_test(k=0):
        if k==len(inputs):	# base case: assignments = dict of input variables with assigned values
            assigned_values = ''.join([tfbits[assignments[x]] for x in inputs])
            vala = eqa.subs(assignments)
            if just_truth:
                table[assigned_values] = vala
                return
            valb = eqb.subs(assignments)
            if vala==valb:
                status = "Equal"
            else:
                status = "UNequal"
            if verbose:
                print "[check_equivalence] %s eqa=%s, eqb=%s on [%s]" % (status, vala, valb, assigned_values)
            table[assigned_values] = (vala==valb)
            return

        if inputs[k] in fixed_inputs:		# fixed
            assignments[inputs[k]] = fixed_inputs[inputs[k]]
            return recursive_test(k+1)
        
        for ival in [True, False]:		# binary partition 
            assignments[inputs[k]] = ival
            recursive_test(k+1)

    recursive_test()
    if just_truth:
        return table

    equiv = all(table.values())
    if equiv:
        status = "ARE"
    else:
        status = "are NOT"
    if verbose:
        print "[check_equivalence] %s and %s %s equivalent" % (eqa, eqb, status)
    return equiv, table

def test_check_equiv1():
    equiv, table = check_equivalence(cnot('a','b').eq, swapgate('a', 'b').eq, ['a', 'b'], verbose=True)
    assert not equiv
    
def test_check_equiv2():
    (fe, outputs) = test_funceval8()
    equiv, table = check_equivalence(fe.result, swapgate('a', 'b'), 'ab', verbose=True)
    assert equiv

    equiv, table = check_equivalence(fe.result, cnot('a', 'b'), 'ab', verbose=True)
    assert not equiv

def test_check_equiv3():
    # fredkin made from toffolis
    fe = FuncEval(all_functions, verbose=True)
    inputs = "abc"
    z = fe.eval_composition("""
		toffoli(a,b,c)
		toffoli(a,c,b)
		toffoli(a,b,c)
    """, inputs)
    equiv, table = check_equivalence(fe.result, fredkin('a', 'b', 'c'), 'abc', verbose=True)
    assert equiv
    return equiv, table

#-----------------------------------------------------------------------------
# class for a Circuit, with block inputs and outputs

class Circuit(MultipleCircuitElement):
    '''
    Representation of a circuit with block inputs and outputs.
    Parses simple language to create MultipleCircuitElement 

    Examples:

    inputs x,y,z			// define the inputs
    outputs m				// define the outputs
    m = nand(nand(x,y),nand(y,z))	// funny gate

    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    fun(x,y,z) = nand(nand(x,y),nand(y,z))	// funny gate
    o1 = fun(nand(a,d),nand(a,e),nand(a,f))	// interleaved inputs
    o2 = fun(nand(b,d),nand(b,e),nand(b,f))
    o3 = fun(nand(c,d),nand(c,e),nand(c,f))

    '''
    nodename = 'circuit'

    def __init__(self, cprog, debug=False, functions=None):
        '''
        cprog is a circuit program, given as a string with one command on each line

        functions = dict of allowed known functions; defaults to [and, not, or] if not provided
        '''
        global SEQ		# YUCK - this should be fixed
        global FAULTS
        SEQ[0] = 0		# reset sequence counter
        FAULTS = {}		# reset faults list

        functions = functions or {'and': andgate,
                                  'or': orgate,
                                  'not': notgate,
                                  }
        self.feval = FuncEval(functions, verbose=debug)

        if type(cprog)==unicode:
            cprog = str(MathUnicodeToAscii(cprog)).strip()
        self.cprog = cprog
        pd = self.parse_prog(cprog)
        self.parsedat = pd
        if debug:
            print "parsed program = ", pd
        self.debug = debug
        super(Circuit,self).__init__(*pd['input_args'],outputs=pd['output_args'])

    def make_circ(self):
        '''
        Make circuit for each output from the parsed data circuit operations ('cops')
        '''
        cops = self.parsedat['cops']

        # initialize environment with input and output symbols
        # TODO: error checking, eg no output symbol can be an input symbol?
        env = dict([ (str(x),x) for x in sympy.flatten(self.args) ])
        outenv = dict([ (str(x),x) for x in sympy.flatten(self.outputs) ])
        env.update( outenv )
        basic_gates = [nand,andgate,orgate,notgate]
        for g in basic_gates: env[g.gatename] = g
        outset = {}

        self.env = env

        # walk through cops and construct circuit
        for op in cops:
            lhs = op['lhs']
            rhs = op['rhs']
            # rhs = rhs.replace('nand','Nand')	# grandfathered

            # 1. evaluate rhs
            if rhs.startswith('[') and rhs.endswith(']') and ';' in rhs:	# this is a multiline expression
                # handle multiline expressions, eg [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]
                tmpenv = {}
                for rhsline in rhs[1:-1].split(';'):
                    if '=' in rhsline:
                        (tmp_lhs, tmp_rhs) = rhsline.split('=')
                        tmp_lhs = tmp_lhs.strip()
                        tmp_rhs = tmp_rhs.strip()
                    else:
                        tmp_lhs = None
                        tmp_rhs = rhsline.strip()
                    try:
                        result = self.feval.eval(tmp_rhs, tmpenv)
                    except Exception as err:
                        raise Exception('[Circuit]: Error "%s" evaluating line %d = "%s"' % (err,op['linenum'],op['line']))
                    if tmp_lhs:
                        tmpenv[tmp_lhs] = result
                func_ret = result
            else:
                try:
                    func_ret = self.feval.eval(rhs, env)
                except Exception, err:
                    raise Exception,'[Circuit]: Error "%s" evaluating line %d = "%s", env=%s' % (err,op['linenum'],op['line'], env)

            # 2. assign to lhs
            if lhs in outenv:	# defines an output
                outsym = outenv[lhs]
                if outsym in outset:
                    raise Exception, '[Circuit]: output already defined!  Cannot evaluate line %d = "%s"' % (op['linenum'],
                                                                                                             op['line'])
                outset[outsym] = func_ret

            else:		# it is a new macro or signal definition
                m = re.match('([^\(\)]+)(\(.*\))', lhs)	# parens? then it is a macro
                if m:
                    # handle defining a macro - basically a new function to add to the function evaluator

                    opname = m.group(1).strip()
                    opargs = m.group(2)[1:-1].split(',')
                    if self.debug:
                        print "op=%s, opname=%s, opargs=%s" % (op, opname, opargs)
                    if opname in env:
                        raise Exception,'[Circuit]: cannot redefine macro %s in line %d = "%s"' % (opname,op['linenum'],op['line'])
                    if '__' in rhs:
                        raise Exception,'[Circuit]: syntax error on RHS in line %d = "%s"' % (op['linenum'],op['line'])

                    def newfunc(*args):
                        if not len(args)==len(opargs):
                            raise Exception("macro function %s called with %d arguments, expecting %s, for lhs=%s in %s" % (opname,
                                                                                                                            len(self.args),
                                                                                                                            len(opargs),
                                                                                                                            lhs,
                                                                                                                            op['line'],
                            ))
                        args = list(args)
                        for k in range(len(args)):
                            if isinstance(args[k], Signal):
                                args[k] = args[k].val
                        argmap = dict(zip(opargs, args))
                        if self.debug:
                            print "macro %s: applying to argmap=%s, func_ret=%s" % (opname, argmap, func_ret)
                            print "args[0]=%s, type=%s" % (args[0], type(args[0]))
                        return func_ret.eq.subs(argmap)

                    env[opname] = newfunc
                    self.feval.add_function(opname, newfunc)

                    continue

                m = re.match('([^\(\)]+)$',lhs)	# no parens? then it is a signal definition
                if m:
                    # handle defining a signal
                    opname = m.group(1).strip()
                    if opname in env:
                        raise Exception,'[Circuit]: cannot redefine signal definition %s in line %d = "%s"' % (opname,op['linenum'],
                                                                                                               op['line'])
                    env[opname] = func_ret
                    continue

                raise Exception,'[Circuit]: syntax error on LHS of line %d = "%s"' % (op['linenum'],op['line'])

        # make sure all outputs are defined
        for x in sympy.flatten(self.outputs):
            if x not in outset:
                raise Exception,'[Circuit]: output %s is undefined' % x

        self.circ = [ outset[x] for x in sympy.flatten(self.outputs) ]
                        
    def parse_prog(self,cprog):
        linenum = 0
        cops = []
        input_args = output_args = None
        for k in cprog.split('\n'):
            linenum += 1
            k = k.strip()
            if k=='': continue
            if k[0:2]=='//':		# line is pure comment
                continue

            def getargs(x):
                if '//' in x: (x,comment) = x.split('//',1)
                return x.strip()

            def parse_args(x):	# turn string into suitable arguments for CircuitElement
                x = x.replace(' ','').replace('\t','')
                if '[' in x: argset = x.split('],[')	# it's a block input
                else: argset = x.split(',')		# no blocks
                def mklist(x):
                    if ('[' in x or ']' in x) : return [y.strip() for y in x.replace(']','').replace('[','').split(',')]
                    return x
                return [ mklist(x.strip()) for x in argset]

            m = re.match('inputs (.*)',k)	
            if m:
                input_args = parse_args(getargs(m.group(1)))
                continue

            m = re.match('outputs (.*)',k)
            if m:
                output_args = parse_args(getargs(m.group(1)))
                continue
                
            m = re.match('([^=;/]+)=(.*)',k)
            if m:
                if not (input_args and output_args):
                    raise Exception,'[Circuit]: inputs & outputs must be defined before statemens; line %s = "%s"' % (linenum,k)
                lhs = getargs(m.group(1))
                rhs = getargs(m.group(2))
                cops.append({'lhs':lhs,'rhs':rhs,'linenum':linenum,'line':k})
                continue

            raise Exception,'[Circuit]: cannot parse line %s = "%s"' % (linenum,k)

        parsedat = {'input_args':input_args,
                    'output_args':output_args,
                    'cops': cops,
                    }
        return parsedat

#------------------------------

def test_simp_circ1():
    x = Circuit('''
    inputs x,y
    outputs m
    m = and(x,y)
    ''')
    print x
    # assert(False)
    assert(len(x.circ)==1)
    assert(x.eval(1,1)==[True])
    assert(x.eval(0,1)==[False])
    assert(x.eval(1,0)==[0])
    assert(x.eval(0,0)==[False])
    return x

def test_circ1():
    x = Circuit('''
    inputs x,y,z			// define the inputs
    outputs m				// define the outputs
    m = nand(nand(x,y),nand(y,z))	// majority vote gate
    ''', functions=all_functions)
    print x
    # assert(False)
    assert(len(x.circ)==1)
    assert(x.eval(1,1,1)==[True])
    assert(x.eval(0,1,1)==[True])
    assert(x.eval(0,1,0)==[0])
    assert(x.eval(0,0,0)==[False])
    return x

# MACROS BROKEN for now

def x_test_circ2():
    x = Circuit('''
    inputs x,y,z			// define the inputs
    outputs m				// define the outputs
    v(x,y) = nand(x,y)			// test macro
    q = nand(y,z)			// test signal definition
    m = nand(v(x,y),q)			// majority vote gate
    ''', functions=all_functions)
    assert(x.eval(1,1,1)==[True])
    assert(x.eval(0,1,1)==[True])
    assert(x.eval(0,1,0)==[0])
    assert(x.eval(0,0,0)==[False])
    return x

def x_test_circ3():	# bad - duplicated q term
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    q(x,y,z) = nand(nand(x,y),nand(y,z))	// first step in majority vote gate
    maj(x,y,z) = nand(nand(q(x,y,z),q(x,y,z)),nand(x,z))	// majority vote gate
    o1 = maj(nand(a,d),nand(a,e),nand(a,f))	// interleaved inputs
    o2 = maj(nand(b,d),nand(b,e),nand(b,f))
    o3 = maj(nand(c,d),nand(c,e),nand(c,f))
    ''', functions=all_functions)
    return x

def x_test_circ3a():	# test multiline macro
    x = Circuit('''
    inputs [a,b,c]
    outputs m
    maj(x,y,z) = [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]	// majority vote gate
    m = maj(a,b,c)
    ''', functions=all_functions)
    # assert(x.eval([True,True,True])==[True])
    return x

def x_test_circ4():	# good - 353 fault paths
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    maj(x,y,z) = [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]	// majority vote gate
    o1 = maj(nand(a,d),nand(a,e),nand(a,f))	// interleaved inputs
    o2 = maj(nand(b,d),nand(b,e),nand(b,f))
    o3 = maj(nand(c,d),nand(c,e),nand(c,f))
    ''', functions=all_functions, debug=True)
    assert(x.eval([True,True,True],[False,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,True,False])==[False, False, False])
    return x

def x_test_circ5():	# good - 192 fault paths
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    maj(x,y,z) = Or(And(x,z),Or(And(x,y),And(y,z)))	// majority vote gate
    o1 = maj(nand(a,d),nand(a,e),nand(a,f))	// interleaved inputs
    o2 = maj(nand(b,d),nand(b,e),nand(b,f))
    o3 = maj(nand(c,d),nand(c,e),nand(c,f))
    ''', functions=all_functions)
    assert(x.eval([True,True,True],[False,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,True,False])==[False, False, False])
    return x

def test_circ6():
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    o1 = or(or(and(or(not(a), not(d)), or(not(a), not(e))), and(or(not(a), not(d)), or(not(a), not(f)))), and(or(not(a), not(e)), or(not(a), not(f))))
    o2 = or(or(and(or(not(b), not(d)), or(not(b), not(e))), and(or(not(b), not(d)), or(not(b), not(f)))), and(or(not(b), not(e)), or(not(b), not(f))))
    o3 = or(or(and(or(not(c), not(d)), or(not(c), not(e))), and(or(not(c), not(d)), or(not(c), not(f)))), and(or(not(c), not(e)), or(not(c), not(f))))
    ''', functions=all_functions)
    assert(x.eval([True,True,True],[False,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,True,False])==[False, False, False])
    return x

def test_circ7():
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    nad = nand(a,d)
    o1 = or(or(and(nad, nand(a,e)), and(nand(a,d), nand(a,f))), and(nand(a,e), nand(a,f)))
    o2 = or(or(and(nand(b,d), nand(b,e)), and(nand(b,d), nand(b,f))), and(nand(b,e), nand(b,f)))
    o3 = or(or(and(nand(c,d), nand(c,e)), and(nand(c,d), nand(c,f))), and(nand(c,e), nand(c,f)))
    ''', functions=all_functions)
    assert(x.eval([True,True,True],[False,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,True,False])==[False, False, False])
    return x

def test_circ8():	# 192 fault paths
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    nad = nand(a,d)
    nae = nand(a,e)
    naf = nand(a,f)
    nbd = nand(b,d)
    nbe = nand(b,e)
    nbf = nand(b,f)
    ncd = nand(c,d)
    nce = nand(c,e)
    ncf = nand(c,f)
    o1 = or(or(and(nad, nae), and(nad, naf)), and(nae, naf))
    o2 = or(or(and(nbd, nbe), and(nbd, nbf)), and(nbe, nbf))
    o3 = or(or(and(ncd, nce), and(ncd, ncf)), and(nce, ncf))
    ''', functions=all_functions)
    assert(x.eval([True,True,True],[False,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,False,False])==[True, True, True])
    assert(x.eval([True,True,True],[True,True,False])==[False, False, False])
    return x

def x_test_circ9():	# bad - fails for certain inputs
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    maj(x,y,z) = [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]	// majority vote gate
    o1 = maj(nand(a,d),nand(b,e),nand(c,f))	// three identical circuits
    o2 = maj(nand(a,d),nand(b,e),nand(c,f))	
    o3 = maj(nand(a,d),nand(b,e),nand(c,f))	
    ''', functions=all_functions)
    return x

def x_test_circ10():	# bad - fails for certain inputs
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    maj(x,y,z) = [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]	// majority vote gate
    o1 = maj(nand(a,d),nand(a,d),nand(a,d))	// three identical circuits
    o2 = maj(nand(b,e),nand(b,e),nand(b,e))	
    o3 = maj(nand(c,f),nand(c,f),nand(c,f))	
    ''', functions=all_functions)
    return x

#-----------------------------------------------------------------------------

def test_rev_circ1():
    x = Circuit('''
    inputs a,b
    outputs x
    x = xor(a,b)
    ''', functions=all_functions)
    return x

#-----------------------------------------------------------------------------
# compute all fault paths
#
# Take all pairs of NAND gates and see how many lead to multiple faults in the output
# Also check single NAND gate failures (shouldn't cause multiple faults in the output!)

def setup_ftnand_check():
    # n = ftnand(['a','b','c'],['d','e','f'])
    #n = test_circ3()
    n = test_circ4()
    gateidset = list(set(re.findall('\[(n[0-9]+)\]',''.join(n.ident()))))
    gateidset.sort()
    return n,gateidset

def is_block_bad(block):
    errors = ['e%s' % x for x in FAULTS]
    outset = [ y.name for y in list(block.atoms()) ]
    for e in errors:
        if e in outset: return 1
    return 0

def ftnand_check1():
    '''
    Check all single NAND gate failures
    '''
    global FAULTS
    (n,gis) = setup_ftnand_check()

    faultset = {}
    for fault in gis:
        FAULTS = [fault]
        outblocks = n.eval()
        blocksbad = [is_block_bad(x) for x in outblocks]
        isbad = sum(blocksbad)>1
        if isbad:
            faultset[fault] = outblocks
    print "%d fault paths" % len(faultset)
    return faultset

def ftnand_check2():
    '''
    Check all double NAND gate failures.
    The output has to have errors in more than one block for the FT NAND to fail.
    '''
    global FAULTS
    (n,gis) = setup_ftnand_check()

    faultset = {}
    ngis = len(gis)

    for j in range(ngis):
        fault1 = gis[j]
        for k in range(j):
            fault2 = gis[k]
            FAULTS = [fault1, fault2]
            outblocks = n.eval()
            blocksbad = [is_block_bad(x) for x in outblocks]
            isbad = sum(blocksbad)>1
            if isbad:
                faultset[(fault1,fault2)] = outblocks
    print "%d fault paths" % len(faultset)
    return faultset

#-----------------------------------------------------------------------------
# check for correct NAND gate behavior

def check_nand_behavior(ftnand):
    '''
    circ should be a circuit with block inputs and block outputs.
    The circuit is good if the output can be properly decoded, ie with a majority vote.
    All possible good inputs (ie majority ok) are checked.
    '''
    nb = len(ftnand.circ)	# number of bits in each block
    nmaj = int(nb/2)   	# number of bits needed for majority
    if nb&1: nmaj += 1		# if odd

    def bin2list(x,nbits): return [ (b=='1') for b in (bin(x)[2:].rjust(nbits, '0')) ]
    def maj(x): return x.count(True)>=nmaj
    def decode(x): return maj(x)

    # check nand truth table on all possible inputs
    if 0:
        for x in range(2**nb):
            for y in range(2**nb):
                xblock = bin2list(x,nb)
                yblock = bin2list(y,nb)
                expect = not (decode(xblock) and decode(yblock))
                # print "x,y: ",xblock, yblock
                actual = ftnand.eval(xblock,yblock)
                if not (expect==decode(actual)):
                    raise Exception, "Fault!  x=%s, y=%s, circuit output=%s" % (xblock,yblock,actual)

    # check nand truth table on all possible inputs for each block (not both at same time)
    if 1:
        for x in range(2**nb):
            for y in [0,(2**nb)-1]:
                xblock = bin2list(x,nb)
                yblock = bin2list(y,nb)
                expect = not (decode(xblock) and decode(yblock))
                # print "x,y: ",xblock, yblock
                actual = ftnand.eval(xblock,yblock)
                if not (expect==decode(actual)):
                    raise Exception, "Fault!  x=%s, y=%s, circuit output=%s, expecting=%s, nmaj=%d" % (xblock,yblock,actual, expect, nmaj)
                actual = ftnand.eval(yblock,xblock)
                if not (expect==decode(actual)):
                    raise Exception, "Fault!  x=%s, y=%s, circuit output=%s, expecting=%s, nmaj=%d" % (xblock,yblock,actual, expect, nmaj)

    return True
                
#-----------------------------------------------------------------------------
# routines for tutor

def check_ft_threshold(expect, ans, options=None):
    '''
    ans = circuit specification
    compute threshold, and return that along with drawing of circuit
    '''
    msg = '<html>'
    try:
        x = Circuit(ans)
    except Exception,err:
        return {'ok':False,'msg':'<p>Failed in creating circuit from your input<p><font color=red>%s</font>' % err}

    try:
        svg = x.plot()
    except Exception,err:
        return {'ok':False,'msg':'<p>Failed in drawing your circuit<p>%s' % err}

    # msg += '<p>Your input:<pre>%s</pre></p>' % x
    msg += '<p>Successfully created circuit from your input:'
    msg += '<center>%s</center></p>' % svg
    if 1:
        msg += '<p>Outputs given by: <pre>%s</pre></p>\n' % x.circ
        msg += '<p>Boolean equations describing outputs are: </p><p><ul>'
        for k in x.eq:
            msg += '<li>%s</li>' % k
        msg += '</ul></p>'

    try:
        check_nand_behavior(x)
        is_ok = True
    except Exception,err:
        msg += '<p>Warning: Your circuit is not a legitimate NAND gate acting on all possible tripple-signal blocks.<p><font color=red>%s</font></text>' % err
        is_ok = False
        # return {'ok':False,'msg':msg}

    try:
        # ftp = x.ft_check2()
        ftp = x.ft_check3(do_nand=True)
    except Exception,err:
        msg += '<p>Failed in counting fault paths! Error=%s</p></text>' % err
        return {'ok':False,'msg':msg}
        
    msg += "<p><hr width='40%%'></hr>Your circuit has %d fault paths (threshold=%f)</p>\n" % (len(ftp),1.0/len(ftp))
    msg += '''<p>Fault paths are:  [<a onclick="javascript:$('#fpaths').toggle();" href="javascript:void(0);" id="fpaths_sh">show</a>]<br/>\n'''
    msg += "<ul id='fpaths' style='display:none'>\n"
    for k in ftp:
        msg += '<li><b>%s</b>:%s</li>\n' % (repr(k),str(ftp[k]))
    msg += "</ul>\n"
    msg += "</p>\n"

    msg += '</html>'

    return {'ok':is_ok,'msg':msg}

#-----------------------------------------------------------------------------
# unit tests

def x_test_gviz1():
    dots = """
        digraph circ
        {
        xin [label="<a>a|<b>b|<c>c" shape=record];
        yin [label="<d>d|<e>e|<f>f" shape=record];
        "xin":a -> n1;
        "yin":d -> n1;
        "xin":b -> n2;
        "yin":e -> n2;
        "xin":c -> n3;
        "yin":f -> n3;
        }
    """
    g = gviz.AGraph(string=dots)
    sfp = StringIO()
    g.draw(path=sfp, format='svg', prog='dot')
    svg = sfp.getvalue()
    print svg
    # open('test1.svg','w').write(svg)
    #assert('<g id="node3" class="node"><title>n2</title>' in svg)
    assert('<title>n2</title>' in svg)

def x_test_ftcirc1():
    circ = ("inputs [a,b,c],[d,e,f]	// define the inputs\n"
            "outputs  [o1,o2,o3]   				// define the  outputs\n"
            "//   majority vote gate\n"
            "maj(x,y,z)  =  [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]\n"
            "o1  =  maj(nand(a,d),nand(a,d),nand(a,d))   	//  three identical circuits\n"
            "o2  =  maj(nand(b,e),nand(b,e),nand(b,e))\n"
            "o3  =  maj(nand(c,f),nand(c,f),nand(c,f))\n"
        )
    print circ
    ret = check_ft_threshold('', circ)
    print ret['msg']
    assert('<?xml version="1.0" encoding="UTF-8" standalone="no"?>' not in ret['msg'])
    #assert('Your circuit has 167 fault paths' in ret['msg'])
    assert('Your circuit has 192 fault paths' in ret['msg'])

    msg = ret['msg']
    open('test0.msg','w').write(msg)
    xml = etree.fromstring(msg)

    assert(ret['ok'])

def x_test_circ_plot1():
    x = Circuit('''
    inputs [a,b,c],[d,e,f]			// define the inputs
    outputs [o1,o2,o3]				// define the outputs
    maj(x,y,z) = [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ]	// majority vote gate
    o1 = maj(nand(a,d),nand(a,e),nand(a,f))	// interleaved inputs
    o2 = maj(nand(b,d),nand(b,e),nand(b,f))
    o3 = maj(nand(c,d),nand(c,e),nand(c,f))
    ''')
    svg = x.plot()
    print svg
    open('test3.svg','w').write(svg)
    # assert(False)
    return svg

def x_test_circ_nand1():
    x = Circuit('''
    inputs [a,b,c],[d,e,f]	// define the inputs
    outputs  [o1,o2,o3]   				// define the  outputs
    maj(x,y,z)  =  [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ] //   majority vote gate
    o1  =  maj(nand(a,d),nand(a,d),nand(a,d))   	//  three identical circuits
    o2  =  maj(nand(b,e),nand(b,e),nand(b,e))
    o3  =  maj(nand(c,f),nand(c,f),nand(c,f))
    ''')
    ret = check_nand_behavior(x)
    assert(ret)

def x_test_ftcirc_pascal1():
    circ = '''
            inputs [a,b,c],[d,e,f]	// define the inputs
            outputs  [o1,o2,o3]   				// define the  outputs
            o1 = nand(a,or(and(d,or(e,f)),and(e,f)))
            o2 = nand(b,or(and(d,or(e,f)),and(e,f)))
            o3 = nand(c,or(and(d,or(e,f)),and(e,f)))
            '''
    print circ
    c = Circuit(circ, debug=True)
    print c
    ret = check_ft_threshold('', circ)
    print ret['msg']
    assert('<?xml version="1.0" encoding="UTF-8" standalone="no"?>' not in ret['msg'])
    msg = ret['msg']
    open('test0.msg','w').write(msg)
    xml = etree.fromstring(msg)

    assert(ret['ok'])
    assert('Your circuit has 66 fault paths' in msg)
    return(msg)

def x_test_ftcirc_pascal2():
    '''
    Circuit from Pascal Notz <pnotz@MIT.EDU>, July 2014
    '''
    circ = '''
            inputs [a,b,c],[d,e,f]	// define the inputs
            outputs  [o1,o2,o3]   				// define the  outputs
            o1 = nand(a,or(and(d,or(e,f)),and(e,f)))
            o2 = nand(b,or(and(d,or(e,f)),and(e,f)))
            o3 = nand(c,or(and(d,or(e,f)),and(e,f)))
            '''
    def invar(x, vn):
        return { sympy.Symbol(s): x for s in vn }

    def invar2(x, y):
        vabc = invar(x, 'abc')
        vdef = invar(y, 'def')
        vabc.update(vdef)
        return vabc

    c = Circuit(circ)
    print c.eq

    global FAULTS
    FAULTS =  {sympy.Symbol('a'):None, sympy.Symbol('e'):None }	# this double fault should NOT leave error
    outblocks = c.eval()

    print "outblocks=", outblocks
    values = [(True, True), (True, False), (False, True), (False, False)]
    for v in values:
        vs = invar2(*v)
        blocksbad = [c.is_eq_faulty2(x, varset=vs) for x in outblocks]
        isbad = sum(blocksbad)>1
        print "blocksbad=", blocksbad
        c.blocksbad = blocksbad
        c.varset = vs
        assert(not isbad)
    return c

# test to see if MAJ gate properly inhibits propagation of errors

def x_test_maj_err1():	# BROKEN
    x = Circuit('''
    inputs [a,b,c]
    outputs  [o1]
    maj(x,y,z)  =  [ q = nand(nand(x,y),nand(y,z)); nand(nand(q,q),nand(x,z)) ] //   majority vote gate
    o1  =  maj(a,b,c)
    ''')
    beq = x.eq[0]
    assert(not x.is_eq_faulty2(beq, the_faults={}))

    global FAULTS
    FAULTS =  {sympy.Symbol('a'):None}	    # single fault in maj gate: should leave no error
    x.eval()    
    # do error checking based on all inputs equal
    def invar(x):
        return { sympy.Symbol(s): x for s in 'abc' }
    assert(not x.is_eq_faulty2(x.eval()[0], varset=invar(False)))
    assert(not x.is_eq_faulty2(x.eval()[0], varset=invar(True)))

    FAULTS =  {sympy.Symbol('a'):None, sympy.Symbol('c'):None }	# double fault should leave error
    x.eval()    
    assert(x.is_eq_faulty2(x.eval()[0], varset=invar(False)))
    assert(x.is_eq_faulty2(x.eval()[0], varset=invar(True)))
    return x

def x_test_ftcirc_pascal3():
    '''
    And3 = one of the and(e,f) gates who's output doesn't mix with b.
    (And3, b) should NOT be a fault path
    '''
    circ = '''
            inputs [a,b,c],[d,e,f]	// define the inputs
            outputs  [o1,o2,o3]   				// define the  outputs
            o1 = nand(a,or(and(d,or(e,f)),and(e,f)))
            o2 = nand(b,or(and(d,or(e,f)),and(e,f)))
            o3 = nand(c,or(and(d,or(e,f)),and(e,f)))
            '''
    print circ
    c = Circuit(circ, debug=True)
    print c

    assert(c.circ[0][1][1].id=='And3')

    def invar(x, vn):
        return { sympy.Symbol(s): x for s in vn }

    def invar2(x, y):
        vabc = invar(x, 'abc')
        vdef = invar(y, 'def')
        vabc.update(vdef)
        return vabc

    global FAULTS
    FAULTS =  {'And3':None, sympy.Symbol('b'):None }	# this double fault should NOT leave error
    outblocks = c.eval()

    print "outblocks=", outblocks
    values = [(True, False)]
    for v in values:
        vs = invar2(*v)
        blocksbad = [c.is_eq_faulty2(x, varset=vs) for x in outblocks]
        c.vs = vs
        isbad = sum(blocksbad)>1
        print "blocksbad=", blocksbad
        c.blocksbad = blocksbad
        c.varset = vs
        assert(not isbad)

    return c

#-----------------------------------------------------------------------------
# plot reversible classical circuit

class SceneSVG:
    def __init__(self,name="svg",height=400,width=400):
        self.name = name
        self.items = []
        self.height = height
        self.width = width
        return

    def add(self,item): self.items.append(item)

    def strarray(self):
        var = ['<svg  xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height=\"%d\" width=\"%d\" >\n' % (self.height,self.width),
               " <g style=\"fill-opacity:1.0; stroke:black;",
               "  stroke-width:1;\" >\n"]
        for item in self.items: var += item.strarray()            
        var += [" </g>\n</svg>\n"]
        return var

    def write_svg(self,filename=None):
        if filename:
            self.svgname = filename
        else:
            self.svgname = self.name + ".svg"
        file = open(self.svgname,'w')
        file.writelines(self.strarray())
        file.close()
        return

    def __str__(self):
        '''
        Return SVG as string
        '''
        return ''.join(self.strarray())

    def display(self,prog='open -a firefox'):
        os.system("%s %s" % (prog,self.svgname))
        return        

class Line:
    def __init__(self,start,end,color,width):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        return

    def strarray(self):
        return ["  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" style=\"stroke:%s;stroke-width:%d\"></line>\n" %\
                (self.start[0],self.start[1],self.end[0],self.end[1],colorstr(self.color),self.width)]

class Circle:
    def __init__(self,center,radius,fill_color,line_color,line_width):
        self.center = center
        self.radius = radius
        self.fill_color = fill_color
        self.line_color = line_color
        self.line_width = line_width
        return

    def strarray(self):
        return ["  <circle cx=\"%d\" cy=\"%d\" r=\"%d\"" %\
                (self.center[0],self.center[1],self.radius),
                "    style=\"fill:%s;stroke:%s;stroke-width:%d\"></circle>\n" % (colorstr(self.fill_color),colorstr(self.line_color),self.line_width)]

class Ellipse:
    def __init__(self,center,radius_x,radius_y,fill_color,line_color,line_width):
        self.center = center
        self.radiusx = radius_x
        self.radiusy = radius_y
        self.fill_color = fill_color
        self.line_color = line_color
        self.line_width = line_width
    def strarray(self):
        return ["  <ellipse cx=\"%d\" cy=\"%d\" rx=\"%d\" ry=\"%d\"" %\
                (self.center[0],self.center[1],self.radius_x,self.radius_y),
                "    style=\"fill:%s;stroke:%s;stroke-width:%d\"></ellipse>\n" % (colorstr(self.fill_color),colorstr(self.line_color),self.line_width)]

class Polygon:
    def __init__(self,points,fill_color,line_color,line_width):
        self.points = points
        self.fill_color = fill_color
        self.line_color = line_color
        self.line_width = line_width
    def strarray(self):
        polygon="<polygon points=\""
        for point in self.points:
            polygon+=" %d,%d" % (point[0],point[1])
        return [polygon,\
               "\" \nstyle=\"fill:%s;stroke:%s;stroke-width:%d\"></polygon>\n" %\
               (colorstr(self.fill_color),colorstr(self.line_color),self.line_width)]

class Rectangle:
    def __init__(self,origin,height,width,fill_color,line_color,line_width):
        self.origin = origin
        self.height = height
        self.width = width
        self.fill_color = fill_color
        self.line_color = line_color
        self.line_width = line_width
        return

    def strarray(self):
        return ["  <rect x=\"%d\" y=\"%d\" height=\"%d\"" %\
                (self.origin[0],self.origin[1],self.height),
                "    width=\"%d\" style=\"fill:%s;stroke:%s;stroke-width:%d\"></rect>\n" %\
                (self.width,colorstr(self.fill_color),colorstr(self.line_color),self.line_width)]

class Text:
    def __init__(self,origin,text,size,color):
        self.origin = origin
        self.text = text
        self.size = size
        self.color = color
        return

    def strarray(self):
        return ["  <text x=\"%d\" y=\"%d\" font-size=\"%d\" style='stroke:%s; text-anchor: middle; dominant-baseline:middle'>\n" %\
                (self.origin[0],self.origin[1],self.size,colorstr(self.color)),
                "   %s\n" % self.text,
                "  </text>\n"]

class ForeignObject:
    def __init__(self, origin, text, size):
        self.origin = origin
        self.text = text
        self.size = size
        return

        # <foreignObject x="100" y="100" width="100" height="100">
        # <div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Times; font-size:15px">
        # \(\displaystyle{x+1\over y-1}\)
        # </div>
        # </foreignObject>

    def strarray(self):
        return ["  <foreignObject x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\">\n" %\
                (self.origin[0],self.origin[1], self.size[0], self.size[1]),
                '<div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Times; font-size:12px; font-color:blue;">\n',
                "   %s\n" % self.text,
                "</div> </foreignObject>\n"]

def colorstr(rgb): 
    if rgb=='none':
        return rgb
    colors = {'black': (0,0,0),
              'white': (255,255,255),
              'red': (255,0,0),
              'green': (0,255,0),
              'blue': (0,0,255),
          }
    if rgb in colors:
        rgb = colors[rgb]
    return "#%x%x%x" % (rgb[0]/16,rgb[1]/16,rgb[2]/16)


def test_svg():
    scene = SceneSVG("test")
    scene.add(Rectangle((100,100),200,200,(0,255,255),(0,0,0),1))
    scene.add(Line((200,200),(200,300),(0,0,0),1))
    scene.add(Line((200,200),(300,200),(0,0,0),1))
    scene.add(Line((200,200),(100,200),(0,0,0),1))
    scene.add(Line((200,200),(200,100),(0,0,0),1))
    scene.add(Circle((200,200),30,(0,0,255),(0,0,0),1))
    scene.add(Circle((200,300),30,(0,255,0),(0,0,0),1))
    scene.add(Circle((300,200),30,(255,0,0),(0,0,0),1))
    scene.add(Circle((100,200),30,(255,255,0),(0,0,0),1))
    scene.add(Circle((200,100),30,(255,0,255),(0,0,0),1))
    scene.add(Text((50,50),"Testing SVG",24,(0,0,0)))
    scene.write_svg()
    assert(os.path.exists('test.svg'))
    assert('Testing SVG' in str(scene))
    # scene.display()

#-----------------------------------------------------------------------------

class CircuitPlotSVG(object):
    """A class for managing an SVG circuit plot."""

    xscale = 40.0
    yscale = 60.0
    fontsize = 20.0
    linewidth = 2.0
    scale = yscale
    control_radius = 0.08 * scale
    not_radius = 0.20 * scale
    box_width = 0.30 * scale
    box_height = 0.30 * scale
    swap_delta = 0.10 * scale

    def __init__(self, circuit, input_names=None, **kwargs):
        '''
        inputs_names = ordered list of names of inputs
        '''
        self.name = "qc"
        self.circuit = circuit
        self.ngates = len(circuit.gates)
        self.nqubits = circuit.nbits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()

        if input_names:
            self._label_wires(input_names)

        self._plot_gates()

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        xscale = self.xscale
        yscale = self.yscale
        wire_grid = numpy.arange(self.nqubits*yscale, 0.0, -yscale, dtype=float)	# y locations
        gate_grid = numpy.arange(0.0,self.ngates*xscale, xscale, dtype=float)	# x locations
        self._wire_grid = wire_grid - 0.5*yscale
        self._gate_grid = gate_grid + xscale

    def _create_figure(self):
        """Create the main svg figure."""
        self.svg = SceneSVG(self.name, width=(self.ngates+1.25)*self.xscale, height=(self.nqubits+0.2)*self.yscale)

    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        xstart = self._gate_grid[0]
        xstop =  self._gate_grid[-1]
        xdata = (xstart-self.xscale, xstop+self.xscale)
        for i in range(self.nqubits):
            ydata = (self._wire_grid[i], self._wire_grid[i])
            line = Line((xdata[0], ydata[0]), (xdata[1], ydata[1]), 'black', self.linewidth)
            self.svg.add(line)

    def _plot_gates(self):
        gates = []
        for i, gate in enumerate(self.circuit.gates):
            gate.plot_gate(self, i)

    def _label_wires(self, labels):
        for k in range(self.nqubits):
            ypos = self._wire_grid[self.nqubits - k -1] - 0.15 * self.yscale
            xpos = self._gate_grid[0] - 0.75 * self.xscale
            self.text_label(xpos, ypos, labels[k])

    def text_label(self, x, y, txt):
        self.svg.add(Text([x,y], txt, size=self.fontsize, color='blue'))

    def one_qubit_box(self, t, gate_idx, wire_idx, use_mathjax=False, width=1):
        """Draw a box for a single qubit gate."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        dx = self.box_width*width
        dy = self.box_height
        self.svg.add(Rectangle([x-dx, y-dy], 2*dy, 2*dx, 'white', 'black', self.linewidth))
        if not use_mathjax:
            t = t[1:-1]
            self.svg.add(Text([x,y], t, size=self.fontsize, color='blue'))
        else:
            self.svg.add(ForeignObject([x-dx+2,y-dy+5],t, [2*dx, 2*dy]))

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        radius = self.not_radius
        self.svg.add(Line((xdata[0], ydata[0]), (xdata[1], ydata[1]), 'black', self.linewidth))

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.control_radius
        c = Circle((x,y), radius, 'blue', 'black', self.linewidth)
        self.svg.add(c)

    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.not_radius
        self.svg.add(Circle((x,y), radius, 'none', 'black', self.linewidth))
        self.svg.add(Line((x,y-radius), (x,y+radius), 'black', self.linewidth))
        self.svg.add(Line((x-radius,y), (x+radius,y), 'black', self.linewidth))

    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        d = self.swap_delta
        self.svg.add(Line((x-d,y-d), (x+d,y+d), 'black', self.linewidth))
        self.svg.add(Line((x-d,y+d), (x+d,y-d), 'black', self.linewidth))

def test_circuit_plot1():
    fe = FuncEval(all_functions, verbose=True)
    inputs = OrderedDict([(x,  Signal(x)) for x in "ab"])
    z = fe.eval_composition("""
		cnot(a,b)
		cnot(b,a)
		cnot(a,b)
    """, inputs)
    cp = CircuitPlotSVG(fe.circuit)
    ofn = "test1.svg"
    cp.svg.write_svg(ofn)
    print "wrote %s" % ofn
    
#=============================================================================
# capa customresponse routines for course problems

#-----------------------------------------------------------------------------
# 243 swap

def two_four_three_swap():
    fe = FuncEval(all_functions, verbose=False)
    inputs = "abcd"
    z = fe.eval_composition("""
		fredkin(a,c,d)
		not(a)
		fredkin(a,b,d)
                not(a)
    """, inputs)
    return fe, z
    
def test_243_swap1():
    (fe, z) = two_four_three_swap()
    assert(fe.result.subs({'a':True, 'b':False, 'c':False, 'd':True})==[True, False, True, False])
    assert(fe.result.subs({'a':False, 'b':False, 'c':False, 'd':True})==[False, True, False, False])

def check_243_swap(expect, ans, options=None):
    '''
    ans = circuit specification
    see if the circuit provided matches the 243 swap circuit
    '''
    ans = str(ans)
    fe = FuncEval(reversible_functions, verbose=False)
    inputs = "abcd"
    try:
        z = fe.eval_composition(ans, inputs)
    except Exception as err:
        return {'ok': False, 'msg': 'Error %s parsing your circuit' % str(err)}

    try:
        svg = fe.circuit.plot(xscale=70, box_width=0.50*60)
    except Exception, err: 
        return{'ok':False,'msg': 'Error %s drawing your circuit %s' % (err,ans)}

    msg = '<p>Graphical rendition of your circuit :<center>%s</center>' % (svg)
    
    fe_243, _ = two_four_three_swap()

    equiv, table = check_equivalence(fe.result, fe_243.result, 'abcd', verbose=True)
    ok = equiv
    return {'ok': ok, 'msg': msg}

def test_check_243_swap1():
    ans = """not(a)
fredkin(a,b,d)
not(a)
fredkin(a,c,d)
    """
    ret = check_243_swap(None, ans)
    assert ret['ok']
    return ret

def test_check_243_swap2():
    ans = """not(a)
fredkin(a,b,d)
fredkin(a,c,d)
    """
    ret = check_243_swap(None, ans)
    assert not ret['ok']
    return ret

#----------------------------------------

def make_ccswap():
    fe = FuncEval(all_functions, verbose=True)
    inputs = "abcde"
    z = fe.eval_composition("""
		fredkin(a,b,e)
		fredkin(e,c,d)
                fredkin(a,b,e)
    """, inputs)
    return fe, z

def check_ccswap(expect, ans, options=None):
    '''
    controlled-controlled-swap problem

    ans = circuit specification
    see if the circuit provided matches the ccswap circuit
    '''
    ans = str(ans)
    fe = FuncEval(reversible_functions, verbose=True)
    inputs = "abcde"
    try:
        z = fe.eval_composition(ans, inputs)
    except Exception as err:
        return {'ok': False, 'msg': 'Error %s parsing your circuit' % str(err)}

    try:
        svg = fe.circuit.plot(xscale=70, box_width=0.50*60, labels=['a', 'b', 'c', 'd', 'e=0'])
    except Exception, err: 
        raise
        return{'ok':False,'msg': 'Error %s drawing your circuit %s' % (err,ans)}

    msg = '<p>Graphical rendition of your circuit :<center>%s</center>' % (svg)
    
    fe_ccs, _ = make_ccswap()

    equiv, table = check_equivalence(fe.result, fe_ccs.result, 'abcde', fixed_inputs={'e':False}, verbose=True)
    ok = equiv
    return {'ok': ok, 'msg': msg}

def test_check_ccswap1():
    ans = """
                swap(a,b)
		fredkin(a,b,e)
		fredkin(e,c,d)
                fredkin(a,b,e)
                swap(a,b)
    """
    ret = check_ccswap(None, ans)
    assert ret['ok']
    return ret

#-----------------------------------------------------------------------------
# demux2 problem

def make_demux2():
    fe = FuncEval(all_functions, verbose=True)
    inputs = "abcd"
    z = fe.eval_composition("""
		toffoli(a,b,d)
		not(a)
                toffoli(a,b,c)
                not(a)
                cnot(d,b)
                cnot(c,b)
                not(d)
                toffoli(a,d,b)
                not(d)
                not(a)
                cnot(c,a)
    """, inputs)
    return fe, z
    
def test_make_demux2():
    fe, _ = make_demux2()
    table = check_equivalence(fe.result, None, 'abcd', fixed_inputs={'c':False, 'd':False}, verbose=True, just_truth=True)
    print table
    assert table['0000']==[True, False, False, False]
    assert table['0100']==[False, False, True, False]
    assert table['1000']==[False, True, False, False]
    assert table['1100']==[False, False, False, True]
    return table
    
def test_make_demux2_comp():
    fe, _ = make_demux2()
    d2 = demux_two('a', 'b', 'c', 'd')
    equiv, table = check_equivalence(fe.result, d2, 'abcd', fixed_inputs={'c':False, 'd':False}, verbose=True)
    print table
    assert equiv
    return table

def check_demux2(expect, ans, options=None):
    '''
    demultiplex 2 inputs into four-bits

    ans = circuit specification
    see if the provided circuit matches the demux2 circuit
    '''
    ans = str(ans)
    fe = FuncEval(reversible_functions, verbose=False)
    inputs = "abcd"
    try:
        z = fe.eval_composition(ans, inputs)
    except Exception as err:
        return {'ok': False, 'msg': 'Error %s parsing your circuit' % str(err)}

    try:
        svg = fe.circuit.plot(xscale=50, box_width=0.50*100, fontsize=16, labels=['a', 'b', 'c=0', 'd=0'])
    except Exception, err: 
        return{'ok':False,'msg': 'Error %s drawing your circuit %s' % (err,ans)}

    msg = '<p>Graphical rendition of your circuit :<center>%s</center>' % (svg)
    
    d2 = demux_two('a', 'b', 'c', 'd')

    equiv, table = check_equivalence(fe.result, d2, 'abcd', fixed_inputs={'c':False, 'd':False}, verbose=True)
    ok = equiv
    return {'ok': ok, 'msg': msg}

def test_check_demux2():
    ans = """
		toffoli(a,b,d)
		not(a)
                toffoli(a,b,c)
                not(a)
                cnot(c,b)
                cnot(d,b)
                not(d)
                toffoli(a,d,b)
                not(d)
                cnot(c,a)
                not(a)
    """
    ret = check_demux2(None, ans)
    assert ret['ok']
    return ret
