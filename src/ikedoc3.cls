%
% File:   qbook.cls
% Date:   02-Jul-97
% Author: I. Chuang <ike@isl.stanford.edu>
%	  M. Nielsen <mnielsen@tangelo.phys.unm.edu>
%
% Book style format (latex2e document class)
%
%
% $Id: qbook.cls,v 1.31 2000/05/27 20:26:00 ike Exp $
%
% $Log: qbook.cls,v $
% Revision 1.31  2000/05/27 20:26:00  ike
% myitemize
%
% Revision 1.30  2000/05/24 22:25:46  ike
% added \setlength{\arraycolsep}{4pt}% to mattwo
%
% Revision 1.29  2000/05/15 18:30:26  ike
% added bra versions of the Bell basis macros
%
% Revision 1.28  2000/05/15 05:40:25  ike
% no auto-indexing of lemmas
%
% Revision 1.27  2000/04/23 19:03:56  ike
% removed auto-indexing of corollary and chapter problems
%
% Revision 1.26  2000/04/23 16:00:01  ike
% \arXiv
%
% Revision 1.25  2000/04/23 15:46:04  ike
% \mytt
%
% Revision 1.24  2000/04/23 15:45:21  ike
% new html environment for refs
%
% Revision 1.23  2000/04/22 16:34:46  ike
% added \setlength{\arraycolsep}{4pt} to \mattwoc
%
% Revision 1.22  2000/04/18 05:18:25  ike
% add small vspace at top of BOX
%
% Revision 1.21  2000/04/18 05:05:44  ike
% ifCUP for chapterquote
%
% Revision 1.20  2000/04/18 05:03:39  ike
% ifCUP changes
%
% Revision 1.19  2000/04/18 04:50:58  ike
% mylfbox width for CUP
%
% Revision 1.18  2000/04/18 04:45:47  ike
% added noindent to first line in a BOX
%
% Revision 1.17  2000/04/17 23:34:43  ike
% ifCUP stuff
%
% Revision 1.16  2000/04/17 21:50:17  ike
% *** empty log message ***
%
% Revision 1.15  2000/04/17 21:30:41  ike
% cup style switch
%
% Revision 1.14  2000/04/17 21:22:53  ike
% added CUP ifthen
%
% Revision 1.13  2000/04/16 23:28:27  ike
% added upright pi
%
% Revision 1.12  2000/04/16 21:33:13  ike
% changed myeqnarray arraycolsep from 2pt to 1.5pt
%
% Revision 1.11  2000/04/16 20:15:09  ike
% added small space before fbox on rhs of BOX
%
% Revision 1.10  2000/04/16 20:04:03  ike
% added a little space between bars and paragraph on lhs of mylfbox
%
% Revision 1.9  2000/04/16 19:19:44  ike
% increased width of box, added paragraph indents to BOX
%
% Revision 1.8  2000/04/16 01:03:54  ike
% changed myeqnarry spacing from 1pt to 2pt
%
% Revision 1.7  2000/04/13 23:37:35  ike
% changed \mychoose to use standard TeX \choose
%
% Revision 1.6  2000/04/13 19:07:08  ike
% fixed spacing of summary environment/command
%
% Revision 1.5  2000/04/12 20:24:22  ike
% footers: QCI99 -> QCI2000
%
% Revision 1.4  2000/04/12 06:54:08  ike
% name
%
% Revision 1.3  2000/04/12 06:53:47  ike
% added some inputs to end of file; test6 works ok
%
% Revision 1.2  2000/04/12 06:47:29  ike
% latex2e class
%
% Revision 1.1  2000/04/12 06:45:38  ike
% Initial revision
%
% Revision 1.119  2000/04/10 20:11:50  ike
% removed auto-indexing of exercises
%
% Revision 1.118  2000/04/06 05:06:21  ike
% changed rawhtml entry to not be \tiny font
%
% Revision 1.117  2000/04/03 00:30:14  mnielsen
% added \notdiv command to indicates "does not divide".
%
% Revision 1.116  2000/04/02 16:44:01  ike
% changed onpage to use it's own labels to compare against
%
% Revision 1.115  2000/04/01 07:46:56  ike
% added myeqnarray
%
% Revision 1.114  2000/03/30 03:59:25  mnielsen
% updated notex enviroenment
%
% Revision 1.113  2000/01/03 10:10:48  ike
% moved copyright footer to outside of pages
%
% Revision 1.112  2000/01/03 06:36:19  ike
% removed rcsid page marker
%
% Revision 1.111  2000/01/03 06:08:55  ike
% removed thumbnail index
%
% Revision 1.110  1999/12/29 05:27:07  ike
% single spaced now!
%
% Revision 1.109  1999/12/28 07:25:23  ike
% \rand
%
% Revision 1.108  1999/12/28 00:05:09  mnielsen
% \diag
%
% Revision 1.107  1999/12/28 00:04:20  mnielsen
% \spur
%
% Revision 1.106  1999/12/27 07:37:52  ike
% reals -> {\bf R}
%
% Revision 1.105  1999/12/19 08:53:02  ike
% index propositions properly
%
% Revision 1.104  1999/12/16 05:44:00  mnielsen
% updated \Xdef,\Xdefbox,\XDef to remove \marnote
%
% Revision 1.103  1999/12/14 13:49:11  ike
% typo in CSS
%
% Revision 1.102  1999/12/14 04:49:33  mnielsen
% \CSS
%
% Revision 1.101  1999/12/07 02:32:06  ike
% url -> tiny
%
% Revision 1.100  1999/11/30 07:04:07  ike
% two-column, small font bibliography
%
% Revision 1.99  1999/11/30 03:33:11  ike
% added rawhtml environment for url bib entries
%
% Revision 1.98  1999/11/29 09:07:00  ike
% added idxcite
%
% Revision 1.97  1999/11/29 05:58:50  ike
% added bibmessage
%
% Revision 1.96  1999/11/29 02:46:50  ike
% increased spacing of thumbnail index
%
% Revision 1.95  1999/11/28 21:23:33  ike
% changed setstretch to 1.05 to reduce page count...
%
% Revision 1.94  1999/11/28 06:26:31  ike
% typo
%
% Revision 1.93  1999/11/28 03:21:21  mnielsen
% added \real
%
% Revision 1.92  1999/11/27 23:17:40  ike
% added new mybibitem command to automatically index citations
%
% Revision 1.91  1999/11/25 01:58:27  ike
% added onpage
%
% Revision 1.90  1999/11/21 05:01:02  ike
% added Rbar
%
% Revision 1.89  1999/11/16 06:08:09  ike
% added xbar, ybar
%
% Revision 1.88  1999/11/16 06:06:45  ike
% added vecfour
%
% Revision 1.87  1999/11/14 20:39:18  ike
% added Upsilon and \deg
%
% Revision 1.86  1999/11/13 01:27:59  ike
% \g = \gamma
%
% Revision 1.85  1999/11/12 17:10:16  ike
% added \adag etc
%
% Revision 1.84  1999/11/07 18:25:59  ike
% fixed lemma stepcounter -> refstepcounter
%
% Revision 1.83  1999/11/07 17:59:52  ike
% fixed prechapter section headings by adding chaptermark cmd
%
% Revision 1.82  1999/11/04 11:02:20  mnielsen
% Added \phase
%
% Revision 1.81  1999/11/02 20:16:33  mnielsen
% Added \poly command.
%
% Revision 1.80  1999/10/24 00:37:23  ike
% added matthreec
%
% Revision 1.79  1999/10/10 07:29:24  ike
% added some defs for implement.tex
%
% Revision 1.78  1999/10/04 00:25:33  ike
% added \w = \omega
%
% Revision 1.77  1999/09/28 07:01:58  ike
% added hfill in \end{proof}
%
% Revision 1.76  1999/09/21 06:26:26  ike
% fixed typo in notes environment
%
% Revision 1.75  1999/09/21 01:42:12  ike
% also fixed prechapter
%
% Revision 1.74  1999/09/21 01:32:28  ike
% minor change to mychapter
%
% Revision 1.73  1999/09/05 00:53:12  mnielsen
% added \Sch
%
% Revision 1.72  1999/08/26 23:43:17  mnielsen
% updated notes environment.
%
% Revision 1.71  1999/06/06 01:24:31  ike
% changed bell state notation
%
% Revision 1.70  1999/06/05 21:56:04  mnielsen
% update marginal notes so the spacing works more effectively in the index entries.
%
% Revision 1.69  1999/05/31 20:38:05  mnielsen
% updated \XDef
%
% Revision 1.68  1999/05/28 00:12:23  ike
% testing
%
% Revision 1.67  1999/05/27 23:57:09  mnielsen
% added \Xdefbox command.
%
% Revision 1.66  1999/05/11 17:32:31  mnielsen
% yet another minor change to \Xdef, this time for spacing.
%
% Revision 1.65  1999/05/10 06:46:17  mnielsen
% updated \XDef, yet again.
%
% Revision 1.64  1999/04/30 00:39:33  mnielsen
% minor change to \Xdef.
%
% Revision 1.63  1999/04/30 00:39:18  mnielsen
% *** empty log message ***
%
% Revision 1.62  1999/04/30 00:38:49  mnielsen
% minor change to \XDef.
%
% Revision 1.61  1999/04/28 23:09:39  mnielsen
% Added proposition environment.
%
% Revision 1.60  1999/04/28 23:09:06  mnielsen
% *** empty log message ***
%
% Revision 1.59  1999/04/28 07:43:44  ike
% got \XDef command usage wrong.  Should be:
% 	\XDef[optional subentry]{indexentry-also-marnote}
%
% Revision 1.58  1999/04/28 07:10:30  ike
% added \XDef command - usage: \XDef{indexentry-also-marnote}[optional subentry]
%
% Revision 1.57  1999/04/19 03:17:32  ike
% added \def\@currentlabel{\p@problem\theproblem} to exercises &
% problems environments to fix \label problem
%
% Revision 1.56  1999/04/08 03:44:21  mnielsen
% minor changes.
%
% Revision 1.55  1999/04/08 03:42:05  mnielsen
% minor changes.
%
% Revision 1.54  1999/04/08 03:31:02  mnielsen
% more minor changes to \Xdef
%
% Revision 1.53  1999/04/08 03:26:36  mnielsen
% more minor changes to \Xdef
%
% Revision 1.52  1999/04/08 03:25:39  mnielsen
% minor change to \Xdef
%
% Revision 1.51  1999/04/02 22:08:27  mnielsen
% Changed \Xdef so that the indexed entry is automatically put into the margin.
%
% Revision 1.50  1999/03/19 04:09:18  mnielsen
% Updated rest of footers: QIC97 -> QCI99
%
% Revision 1.49  1999/03/17 00:57:51  mnielsen
% Updated footers: QIC97 ->  QCI99
%
% Revision 1.48  1999/03/05 08:24:45  mnielsen
% Modified \prob and \pr.
%
% Revision 1.47  1998/11/01 18:41:03  mnielsen
% added \wt command.
%
% Revision 1.46  1998/10/13 21:42:22  mnielsen
% minor change to the chapterquote environment.
%
% Revision 1.45  1998/09/21 23:04:30  ike
% made BOX .25 in wider
%
% Revision 1.44  1998/09/18 01:13:03  ike
% floated boxes
%
% Revision 1.43  1998/09/12 20:10:21  ike
% typos
%
% Revision 1.42  1998/09/12 18:31:18  ike
% made optional arg for problems and exercise \item - boldface label
%
% Revision 1.41  1998/09/11 20:22:41  ike
% more fixes to exercise environment
%
% Revision 1.40  1998/09/11 20:08:23  ike
% changed exercises, problems environments to use global counter
%
% Revision 1.39  1998/09/11 19:57:36  ike
% made theorem, corollary, lemma auto-indexing
%
% Revision 1.38  1998/09/10 16:42:30  mnielsen
% modified the proof and exercise/problem environments.
%
% Revision 1.37  1998/09/08 20:50:11  ike
% added optional argument for theorem, lemma, corrolary make boldface label
% now inputs ifthen.sty
%
% Revision 1.36  1998/09/07 19:21:47  ike
% noindent mylfbox
%
% Revision 1.35  1998/09/07 19:19:06  ike
% added mylfbox
%
% Revision 1.34  1998/09/07 19:13:45  ike
% nada
%
% Revision 1.33  1998/09/07 18:36:38  ike
% added lfbox
%
% Revision 1.32  1998/09/07 01:41:48  ike
% proof:
%
% Revision 1.31  1998/09/07 01:39:40  ike
% added ~ before \Box in proof
%
% Revision 1.30  1998/09/07 01:07:31  ike
% changed notes to history and further reading
%
% Revision 1.29  1998/09/06 23:36:09  ike
% added bell basis macros
%
% Revision 1.28  1998/09/06 22:05:11  ike
% added "list of boxes" patches to BOX, and patches to book.cls
%
% Revision 1.27  1998/09/02 18:58:47  mnielsen
% updated all the problem-type environments to make them slightly more
% user-friendly.
%
% Revision 1.26  1998/08/30 21:39:17  ike
% changed page headers to fix overlaps
%
% Revision 1.25  1998/08/30 20:09:57  ike
% changed setstretch to 1.1 just to see what it prints out like
%
% Revision 1.24  1998/08/30 19:36:42  ike
% changed vec and mat to use [] instead of ()
%
% Revision 1.23  1998/08/30 19:29:11  ike
% added margin note command \marnote, and fixed even-odd margins for
% twoside
%
% Revision 1.22  1998/08/30 04:48:05  ike
% added marnote
%
% Revision 1.21  1998/08/27 05:39:46  ike
% increased width of BOX by 0.25 in to 5.25
%
% Revision 1.20  1998/08/27 05:38:11  ike
% added usage comment to BOX
%
% Revision 1.19  1998/08/26 22:57:37  mnielsen
% added new "exercise" environment, the singluar of the "exercises" environment,
% for cases when there is only a single exercise.
%
% Revision 1.18  1998/08/26 18:56:50  ike
% added boldface index entry command \Xdef
%
% Revision 1.17  1998/08/25 21:54:55  ike
% same
%
% Revision 1.16  1998/08/25 21:54:38  ike
% fixed Chapterquote
%
% Revision 1.15  1998/08/25 19:02:57  mnielsen
% added a \var command.
%
% Revision 1.14  1998/08/25 04:42:54  ike
% fixed preChapter bug
%
% Revision 1.13  1998/08/25 01:59:02  ike
% fix for myChapter bug
%
% Revision 1.12  1998/08/24 22:24:59  ike
% added summary command
%
% Revision 1.11  1998/08/24 03:17:46  ike
% added prechapter command
%
% Revision 1.10  1998/08/12 06:01:53  ike
% added new BOX command with counter, for text boxes.
% also fixed theorem counter so that one can reference it.
%
% Revision 1.9  1998/03/05 22:53:50  ike
% moved rcsid from head to foot
%
% Revision 1.8  1998/02/18 18:41:21  mnielsen
% more changes to the exercises and problems environments
% .
%
% Revision 1.7  1998/02/17 01:01:28  mnielsen
% made some changes to the exercises and problems environments. I made
% some other changes a few days back; I've forgotten what.
%
% Revision 1.6  1998/01/28 06:40:51  ike
% changed unitlength of blobs to get better thumbnail index
%
% Revision 1.5  1998/01/23 03:08:51  ike
% added cleardoublepage
%
% Revision 1.4  1997/12/15 07:08:59  ike
% fixed log
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load generic book style format & fancy headings style

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ikedoc2}
              [2000/04/11 v1.0
 CUP QC book class]

\usepackage{multicol,doublespace,ifthen}
%\usepackage{multicol,ifthen}
\input{fancyhdr.sty}

%\input{multicol.sty}		% 29-Nov-99 ILC - twocolumn bibliography
\input{article.sty}
%\input{doublespace.sty}
%\input{ifthen.sty}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CUP style?

%\def\CUP{\boolean{false}}
\newcommand{\ifCUP}[2]{#2}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% general page setup

% \headheight 8pt			% no header
\headheight 14pt			% no header
\headsep \headheight

%\setlength{\oddsidemargin}{0.09in}	% 1.25in left margin 
%\setlength{\evensidemargin}{0.09in}	% 1.25in left margin (even pages)

\setlength{\oddsidemargin}{0in}	% 1.25in left margin 
\setlength{\evensidemargin}{0in}	% 1.25in left margin (even pages)
\setlength{\topmargin}{0.0in}		% 1in top margin
\setlength{\textwidth}{6.5in}		% 6.0in text - 1.25in rt margin
\setlength{\textheight}{9in}		% Body ht for 1in margins
%\addtolength{\topmargin}{-\headheight}	% No header, so compensate
%\addtolength{\topmargin}{-\headsep}	% for header height and separation

\addtolength{\topmargin}{-2ex}	% compensate for header
\addtolength{\topmargin}{-5pt}	% compensate for header
\addtolength{\textheight}{-\headheight}	% compensate

\addtolength{\topmargin}{-0.125in}	% make 1'' bottom margin

\if@twoside         			% If two-sided printing.
	\setlength{\evensidemargin}{0in}
	\typeout{MYTHESIS: Two sided margins (even=0, odd=0.25)...}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fancy page headings

%\pageheading{\mbox{\tiny Chuang-Thesis96 \today}}

\fancypagestyle{plain}{		% clear header for plain plages like chapters
\fancyhf{}			% clear h&f 
\fancyfoot[C]{\thepage}
%\fancyfoot[RO]{\mbox{\tiny QCI2000 \copyright \today}}
%\fancyfoot[LE]{\mbox{\tiny QCI2000 \copyright \today}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
}

% 14-Dec-97 ILC: this line is automatically changed by RCS each time a
% file is checked in.  We will redefine it in each file so that the
% pages get labeled according to the current file.
\def\rcsid{$$Id: qbook.cls,v 1.31 2000/05/27 20:26:00 ike Exp $$}

\pagestyle{fancy}		% main defailt page style
\fancyhead[LE,RO]{\sl\small\rightmark}
\fancyhead[LO,RE]{\sl\small\leftmark}
%\fancyhead[LE]{\sl\small\rightmark}
%\fancyhead[RO]{\sl\small\rightmark}
\fancyhead[LO]{}
\fancyhead[RE]{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
% \fancyhead[RE]{\mbox{\tiny{\rcsid}}}
% \fancyhead[LO]{\mbox{\tiny{\rcsid}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uncomment the next four lines to get rcsid  02Jan99 ILC
% 
% \fancyfoot[RE]{\mbox{\tiny{\rcsid}}}
% \fancyfoot[LO]{\mbox{\tiny{\rcsid}}}
% \fancyfoot[RO]{\mbox{\tiny QCI99 \copyright \today}}
% \fancyfoot[LE]{\mbox{\tiny QCI99 \copyright \today}}

% and then comment out the next two:
%
%\fancyfoot[RE]{\mbox{\tiny QCI2000 \copyright \today}}
%\fancyfoot[LO]{\mbox{\tiny QCI2000 \copyright \today}}

\fancyfoot[C]{\thepage}


%\fancypagestyle{mythesis}{	% clear header for plain plages like chapters
%\fancyhead[RE]{}
%\fancyhead[LO]{}
% \fancyhead[RO]{\sl\small\leftmark\rblob}
% \fancyhead[LE]{\lblob\sl\small\rightmark}
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thumbnail index

\setlength{\unitlength}{18mm}
%\newcommand{\blob}{\rule[-.2\unitlength]{2\unitlength}{.5\unitlength}}
\newcommand{\blob}{\rule[-.1\unitlength]{1\unitlength}{.25\unitlength}}
%\newcommand{\blob}{\circle*{0.35}}

%\setlength{\unitlength}{10mm}
\setlength{\unitlength}{20mm}
\newcommand\rblob{%
  \begin{picture}(0,0)
%    \put(1,-\value{chapter}){\blob}
    \put(0.5,-\value{chapter}){\blob}
  \end{picture}}

\newcommand\lblob{%
  \begin{picture}(0,0)
%    \put(-3.5,-\value{chapter}){\blob}
    \put(-1.75,-\value{chapter}){\blob}
  \end{picture}%
  }

% \fancyhead[RE]{\sl\small\leftmark}
% \fancyhead[LO]{\sl\small\leftmark}
% \fancyhead[LE]{\lblob\sl\small\rightmark}

% 02-Jan-00 Uncomment the following two lines to get thumbnail index back
% \fancyhead[RO]{\sl\small\rightmark\rblob}
% \fancyhead[LE]{\lblob\sl\small\leftmark}

\fancyhead[RO]{\sl\small\rightmark}
\fancyhead[LE]{\sl\small\leftmark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% line separation

\setstretch{1.2}
% \setstretch{1.1}
%\setstretch{1.05}	% 28-Nov-99 ILC
%\setstretch{1.00}	% 28-Dec-99 ILC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for index

\newcommand{\X}[1]{\index{#1}}
\newcommand{\Xdef}[1]{\index{#1|bold}}
\newcommand{\Xdefbox}[1]{\index{#1|bold}}
\newcommand{\XDef}[2][foo]{
	\ifthenelse{\equal{#1}{foo}}{
		\index{#2|bold}
	}{
		\index{#2!#1|bold}
	}}
%
%\newcommand{\Xdef}[1]{\index{#1|bold}\marnote{\vspace{-2.2mm}#1}}
%\newcommand{\XDef}[2][foo]{\marnote{\vspace{-2.2mm}#2, #1}% 
%	\ifthenelse{\equal{#1}{foo}}{
%		\index{#2|bold}
%	}{
%		\index{#2!#1|bold}
%	}}
\newcommand{\bold}[1]{{\bf{#1}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new bib method which automatically cites!
% 27nov99 ILC

\def\mybibitem[#1]#2#3{\item[\@biblabel{#1}\hfill]
%	{\typeout{\string\bibcite{#2}{#1}{#3}}}
	\if@filesw{\immediate%
%			{\typeout{hello world}}
		       \write\@auxout{\string\bibcite{#2}{#1#3}}}\fi%
	\ignorespaces}

\newcommand{\idxcite}[2]{#1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 29-Nov-99 ILC: for url entries in bib

%\newenvironment{rawhtml}{\newblock\tiny\tt}{}
\newenvironment{rawhtml}{\newblock\tt}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add message before biblist starts
% 28nov99 ILC

\def\bibmessage{}	% default message = empty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special indented-body paragraphs

\newlength{\mylmoffset}
\newskip\@mytempskip

\def\parbindent#1{
	\setlength{\mylmoffset}{#1}
	\setlength{\parindent}{-\mylmoffset}
	\addtolength{\oddsidemargin}{\mylmoffset}
	\addtolength{\evensidemargin}{\mylmoffset}
	\@mytempskip 0ex \relax
	\advance \@mytempskip -\mylmoffset \relax
	\advance \@mytempskip -\mylmoffset \relax
	%\message{section indentation \@mytempskip}
	%\def\mysection{\hspace*{-\mylmoffset}\section}
	%\def\mysubsection{\hspace*{-\mylmoffset}\subsection}

	\def\mysection{\@startsection {section}{1}
		{\@mytempskip}
		{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}{\Large\bf}}
	\def\mysubsection{\@startsection{subsection}{2}
		{\@mytempskip}
		{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}{\large\bf}}
	}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special indented-body paragraphs - usage

%\parbindent{3.5ex}
%\def\bul{\hspace*{-3.5ex}$\bullet$\ }
%\def\tbul{\hspace*{-3.5ex}\hspace*{-3.5ex}$\bullet$\ }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 12-Jan-93 IC: lfbox is like fbox but puts double lines on left side only

\newdimen\lfboxsep
\lfboxsep = 2pt

\long\def\lfbox#1{\leavevmode\setbox\@tempboxa\hbox{#1}\@tempdima\fboxrule
    \advance\@tempdima \fboxsep \advance\@tempdima \dp\@tempboxa
   \hbox{\lower \@tempdima\hbox
  {\vbox{%\hrule \@height \fboxrule
          \hbox{\vrule \@width \fboxrule \hskip\lfboxsep
		\vrule \@width \fboxrule \hskip\fboxsep
          \vbox{\vskip\fboxsep \box\@tempboxa\vskip\fboxsep}
		\hskip \fboxsep
%		\vrule \@width \fboxrule
		}
%                 \hrule \@height \fboxrule
		}}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure definitions

\newcommand{\mypsfig}[2]{\psfig{file=#1,#2}}

\newcommand{\myfig}[4]{
	\begin{figure}[hbtp]
	\begin{center}
	\mbox{{\mypsfig{#1}{#2}}}
	\end{center}
	\caption{#3}
	\label{fig:#4}
	\end{figure}
}

\newcommand{\myffig}[6]{
	\begin{figure}[hbtp]
	\begin{center}
	\mbox{\mypsfig{#1}{#2}\hspace*{3ex}\mypsfig{#3}{#4}}
	\end{center}
	\caption{#5}
	\label{fig:#6}
	\end{figure}
}

\newcommand{\myffigv}[6]{
	\begin{figure}[hbtp]
	\begin{center}
	\mbox{\mypsfig{#1}{#2}}
	\vspace*{4ex}
	\mbox{\mypsfig{#3}{#4}}
	\end{center}
	\caption{#5}
	\label{fig:#6}
	\end{figure}
}

\newcommand{\inlinepsfig}[2]{\begin{center}
	\mbox{\mypsfig{#1}{#2}}\end{center}}

\def\qbul{\hspace*{2ex}$\bullet$\ }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enviroment shortcut definitions

\newenvironment{ctabular}[1]{\begin{center}\begin{tabular}{#1}}%
	{\end{tabular}\end{center}}

\newenvironment{ctabularline}[1]{\begin{center}%
		\begin{tabular}{#1}\hline\hline}%
	{\hline\hline\end{tabular}\end{center}}

\newcommand{\be}{\begin{equation}}
\newcommand{\ee}{\end{equation}}

\newcommand{\bea}{\begin{eqnarray}}
\newcommand{\eea}{\end{eqnarray}}

\newcommand{\bfg}{\begin{figure}[htbp]}
\newcommand{\efg}{\end{figure}}

\newcommand{\bd}{\begin{description}}
\newcommand{\ed}{\end{description}}

\newcommand{\ben}{\begin{itemize}%
	\setlength{\parsep}{0pt}\setlength{\itemsep}{-1pt}}
\newcommand{\een}{\end{itemize}}

\newcommand{\eitem}{\item[$\Box$]}
\newcommand{\bitem}{\item[$\bullet$]}

\def\btab{\begin{tabbing}}
\def\etab{\end{tabbing}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paragraph formats (in boxes, etc.)

\newcommand{\ptc}[1]{\parbox[t]{3in}{\begin{center}{#1}\end{center}}}

\newlength{\myboxlen}
\setlength{\myboxlen}{\textwidth}
\addtolength{\myboxlen}{-1ex}
\newcommand{\eqbox}[1]{\par
	\noindent\fbox{\parbox[t]{\myboxlen}{#1}}\vspace*{1ex}}
\newcommand{\smalleqbox}[1]{\par
	\noindent\fbox{\parbox[t]{4in}{#1}}\vspace*{1ex}}

\newcommand{\mybox}[1]{\fbox{#1}}

\newlength{\myhalfpw}
\newskip\@xwidth  \global\@xwidth\textwidth \global\divide\@xwidth\tw@
			\global\advance\@xwidth -0.5ex
\setlength{\myhalfpw}{\@xwidth}

%\@xwidth=\divide\textwidth\tw@
%\message{myhalfpw=\myhalfpw, textwidth=\textwidth, xwidth=\@xwidth}
%\newcommand{\twoeq}[2]{\\\parbox{3in}{\be #1\ee}\parbox{3in}{\be #2 \ee}\\}
%\newcommand{\ftwoeq}[2]{\\
%	\fbox{\parbox{3in}{\be #1\ee}\parbox{3in}{\be #2 \ee}}\\}

\newcommand{\twoeq}[2]{\\\parbox{\myhalfpw}{\be #1\ee}
			\parbox{\myhalfpw}{\be #2 \ee}\\}
\newcommand{\ftwoeq}[2]{\\
	\fbox{\parbox{\myhalfpw}{\be #1\ee}\parbox{\myhalfpw}{\be #2 \ee}}\\}

\newlength{\mysmallboxlen}
\setlength{\mysmallboxlen}{\textwidth}
\addtolength{\mysmallboxlen}{-1ex}
\newcommand{\mysmallbox}[1]{
 	\begin{center}\fbox{\vspace*{-0.5ex}\parbox[t]{\mysmallboxlen}
			{#1}\vspace*{-0.5ex}}\end{center}}

%\newcommand{\mylfbox}[1]{\par{\hspace*{10ex}\lfbox{
%		{\singlespace\parbox{5.0in}{\footnotesize#1}}}}\\}

%\newcommand{\mylfbox}[2]{{\lfbox{{\parbox{#1}{#2}}}}\\}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Misc. Symbols

\newcommand{\degc}{\mbox{$^\circ$}C}
\newcommand{\degk}{K}

\newfont{\spfont}{cmr10}
\newcommand{\ang}{\mbox{\spfont{\AA}\normalsize}}

\newcommand{\eq}[1]{(\ref{eq:#1})}
%\newcommand{\syssize}[1]{#1$\times$#1$\times$#1}
\newcommand{\rcite}[1]{\raisebox{3pt}{\footnotesize[#1]}}

\renewcommand{\d}[1]{\mbox{$_{#1}$}}
\renewcommand{\u}[1]{\mbox{$^{#1}$}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lines

\newcommand{\myline}[1]{\begin{center}
		\rule[-0.5ex]{#1}{0.2pt}\end{center}}	% typical #1=60ex

\newcommand{\mydelim}[1]{\vspace*{0.6cm}\myline{#1}\vspace*{0.6cm}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Math: General Calculus

\newcommand{\gam}[1]{\gamma^{#1}}
\def\g\gamma	% 12-Nov-99 ILC

\def\dms{\displaystyle}

\def\dmu{\partial_\mu}		\def\dnu{\partial_\nu}
\def\lb{\left[}			\def\rb{\right]}
\def\lp{\left(}			\def\rp{\right)}
\def\rar{\rightarrow}
\def\del{\nabla}

\def\yhat{\hat{y}}		\def\xhat{\hat{x}}
\def\zhat{\hat{z}}		\def\rhat{\hat{r}}
\def\thhat{\hat{\theta}}	\def\phhat{\hat{\phi}}

\def\cth{\cos\theta}		\def\sth{\sin\theta}

\def\nhat{\hat{n}}		\def\nhatd{\hat{n}\cdot}
\def\nhatx{\hat{n}\times}	
\def\pdt{\partial_t}
\def\delx{\del\times}		\def\deld{\del\cdot}

\def\Av{\vec{A}}		\def\Jv{\vec{J}}
\def\Sv{\vec{S}}		\def\Ev{\vec{E}}
\def\Bv{\vec{B}}		\def\Dv{\vec{D}}
\def\Hv{\vec{H}}

\def\w{\omega}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Math: Quantum Mechanics

\newcommand{\mattwo}[4]{\left[
	\setlength{\arraycolsep}{4pt}%
	\begin{array}{rr}{#1}&{#2}\\{#3}&{#4}\end{array}\right]}
\newcommand{\vectwo}[2]{\left[
	\begin{array}{r}{#1}\\{#2}\end{array}\right]}
\newcommand{\matthree}[9]{\left[
	\begin{array}{rrr}{#1}&{#2}&{#3}\\
			  {#4}&{#5}&{#6}\\
			  {#7}&{#8}&{#9}
			  \end{array}\right]}

\newcommand{\mattwoc}[4]{\left[
	\setlength{\arraycolsep}{4pt}%
	\begin{array}{cc}{#1}&{#2}\\{#3}&{#4}\end{array}\right]}

\newcommand{\vectwoc}[2]{\left[
	\begin{array}{c}{#1}\\{#2}\end{array}\right]}

\newcommand{\ket}[1]{\mbox{$|#1\rangle$}}
\newcommand{\bra}[1]{\mbox{$\langle #1|$}}
\newcommand{\brak}[1]{\mbox{$\langle #1\rangle$}}
\newcommand{\bk}[2]{\mbox{$\langle #1 | #2 \rangle$}}
\newcommand{\braket}[3]{\mbox{$\langle #1 | #2 | #3 \rangle$}}
%\newcommand{\udag}[1]{\mbox{$#1^{\dagger}$}}
\newcommand{\udag}[1]{#1^{\dagger}}

\def\dt{\Delta t}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% More math/IT/Phys defs - 16-Jan-94 ILC

\def\lbL{\lb\rule{0pt}{2.4ex}}
\def\lpL{\left(\rule{0pt}{2.4ex}}
\def\oh{\frac{1}{2}}
\newcommand{\combin}[2]{\left(\begin{array}{c}{#1}\\{#2}\end{array}\right)}

\renewcommand{\matthree}[9]{\left[
	\begin{array}{lll}{#1}&{#2}&{#3}\\
			  {#4}&{#5}&{#6}\\
			  {#7}&{#8}&{#9}
			  \end{array}\right]}

\newcommand{\matthreec}[9]{\left[
	\begin{array}{ccc}{#1}&{#2}&{#3}\\
			  {#4}&{#5}&{#6}\\
			  {#7}&{#8}&{#9}
			  \end{array}\right]}

\def\nlog{\,\mbox{\rm nlog}\,}
\def\cnlog#1{\,\lceil\mbox{\rm nlog}\, #1\rceil}
\def\prob{p}
\def\ep{\epsilon}

\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}
\newcommand{\ceil}[1]{\lceil #1 \rceil}

% 10-Oct-99 ILC
\def\splus{\sigma^+}
\def\sminus{\sigma^-}
\def\ochi{\overline{\chi}}
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File:   mycite.sty
% Date:   07-Dec-90
% Author: I. Chuang <monkey@.mit.edu>
%
%
\newif\ifraisedcite%raises citations [1] above line 
\def\raisedcitations{\raisedcitetrue}
\def\noraisedcitations{\raisedcitefalse}
\raisedcitations% default

\def\@cite#1#2{%
\ifraisedcite\raisebox{3pt}{\footnotesize[{#1\if@tempswa,#2\fi}]}%
\else[{#1\if@tempswa,#2\fi}]\fi}
% \newcommand{\incite}[1]{\noraisedcitations\cite{#1}\raisedcitations}

\def\apnum#1{\@apnum{\@nameuse{c@#1}}} 
\def\@apnum#1{A\number #1}  %% changed 29 Apr 86

\def\apbnum#1{\@apbnum{\@nameuse{c@#1}}} 
\def\@apbnum#1{B\number #1}  %% changed 29 Apr 86

% 
% replace macros
%
% example of how to define optional parameters
%\def\parbox{\@ifnextchar [{\@iparbox}{\@iparbox[c]}}
%\long\def\@iparbox[#1]#2#3{\leavevmode \@pboxswfalse
%   \if #1b\vbox 
%     \else \if #1t\vtop 
%              \else \ifmmode \vcenter 
%                        \else \@pboxswtrue $\vcenter
%                     \fi
%           \fi
%    \fi{\hsize #2\@parboxrestore #3}\if@pboxsw $\fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theorem environment (updated 09-Sep-98 ILC)
% optional argument for theorem, lemma, corrolary make boldface label

%\newenvironment{corollary}{\noindent {\em Corollary. }}{}
%\newcommand{\proof}{\par\noindent {\em Proof:~}}
%\newenvironment{theorem}{{\em Theorem. }}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Footnotes numbered per section

% \fnsymbol{footnote}
% \setcounter{footnote}{2}
% \newcounter{myfootnote}[section]
%\renewcommand{\thefootnote}{\stepcounter{myfootnote}\fnsymbol{myfootnote}}

\@addtoreset{footnote}{section}		% reset footnote counter each section
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

% \renewcommand{\thefootnote}{\mbox{$\dagger$}}	% yeah this is bad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% notes and further reading environments

% \newenvironment{notes}{\noindent {\bf Notes and references}}{}
\newenvironment{notes}{\vspace{3mm} \noindent {\Large \bf History and
further reading} \vspace{3mm}}{}
\newenvironment{further_reading}{\noindent {\bf Further reading}
		\typeout{OLD OLD OLD!! remove further_reading section
		in \thesection!}
		}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defines an environment for writing out programs

\newcounter{linenumber}
\newenvironment{program}[2]{\vspace{4mm} \noindent
        #1 \vspace{2mm} \\ {\bf Description:} #2 \\ \vspace{-4mm}
         \begin{list}{\thelinenumber \hspace{2mm}}
        {\usecounter{linenumber} \setlength{\rightmargin}{\leftmargin}
        \setlength{\itemsep}{-3pt}}
        }{\end{list} }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mnielsen's macros

% number theory
\newcommand{\notdiv}{\!\not{|}\,}

% error correction
\newcommand{\CSS}{\mbox{CSS}}
\newcommand{\wt}{\mbox{wt}}		% weight


\newcommand{\poly}{\mbox{poly}}
\newcommand{\complex}{{\bf C}}

\newcommand{\ptab}{\hspace{3mm}}        % tab in the program environment

\newcommand{\mod}{\mbox{mod}}
\newcommand{\integers}{{\bf Z}}
\newcommand{\reals}{{\bf R}}

%
% names
%
\newcommand{\Schrodinger}{Schr\"odinger}
\newcommand{\Renyi}{Renyi}

%
% quantum
%
\newcommand{\hilbert}{H}
\newcommand{\beqn}{\begin{eqnarray}}
\newcommand{\eeqn}{\end{eqnarray}}
\newcommand{\tr}{\mbox{tr}}
\newcommand{\spur}{\mbox{sp}}
\newcommand{\diag}{\mbox{diag}}
\newcommand{\la}{\langle}
\newcommand{\ra}{\rangle}
\def\<{\langle}
\def\>{\rangle}
\newcommand{\Sch}{\mbox{Sch}}		% symbol for the Schmidt number
\newcommand{\evop}{{\cal E}}  		% symbol for quantum operations
\newcommand{\Pc}{P}           		% projector onto a coding subsapce

\newcommand{\phase}{S}

%
% probability
%
\newcommand{\pr}{p}
\newcommand{\ex}{{\bf E}}
\newcommand{\var}{\mbox{var}}

\newcommand{\qed}{{\bf QED}}

\newcommand{\vecthree}[3]{\left[
	\begin{array}{c}{#1}\\{#2}\\{#3}\end{array}\right]}

%\newcommand{\mychoose}[2]{\left(
%	\begin{array}{c}{#1}\\{#2}\end{array}\right)}

\newcommand{\mychoose}[2]{{#1 \choose #2}}

\def\mysinglespace{\def\baselinestretch{1}\@normalsize}
\def\endmysinglespace{\par}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% chapter headings

\def\Chapterquote{\chapterquote}
\newcommand{\chapterquote}[2]{\vspace*{-2ex}
%	\typeout{foo = #1}
%	\typeout{bar = #2}
		\begin{quote} %
			\ifCUP{	\setlength{\leftmargin}{1.0cm}}{}%
			{\em #1} \\ -- {#2}\end{quote}\vspace*{1ex}}

\newcommand{\draft}[1]{\mbox{\huge\bf #1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%% FIGURE CAPTION MACRO - indents entire caption, reduces font size to 10pt 
%%% and puts "Figure x:" in bold face
%%%
\makeatletter
\long\def\@makecaption#1#2{
        \vskip 10pt
        \setbox\@tempboxa\hbox{\small {\bf #1:} #2}
        \parindent 1.5em  %% How can we use the global value of this???
        \dimen0=\hsize
        \advance\dimen0 by -2\parindent
        \ifdim \wd\@tempboxa >\dimen0
                \hbox to \hsize{
                        \hfil
%                        \parbox{\dimen0}{\def\baselinestretch{1.05}\small
                        \parbox{\dimen0}{\def\baselinestretch{0.95}\footnotesize
                                {\bf #1:} #2
                                %%\unhbox\@tempboxa
                                }
                        \hfil}
        \else \hbox to \hsize{\hfil \box\@tempboxa \hfil}
        \fi}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new chapter command so we know where we are

\def\myChapter{\mychapter}
\newcommand{\mychapter}[1]{  
  \cleardoublepage
  \chapter{#1}
  \def\mychaptitle{#1}
%  \addtocontents{lof}{\protect\addvspace{10\p@}}%
  \addtocontents{lob}{\protect\addvspace{10pt}}%
  \typeout{                                                           }
  \typeout{<************** CHAPTER \thechapter: #1 *******************>}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% summary environment

\newcommand{\summary}[1]{
	\vspace{1cm}
	\begin{center}
	\fbox{\parbox{15cm}{\vspace*{1ex}~
	\begin{center} {\bf Summary of Chapter \thechapter: \mychaptitle} 
	\end{center}\begin{itemize}
	#1
	\end{itemize}}~~~~}\end{center}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% marginal notes

\reversemarginpar			% notations go near spine
\newcommand{\marnote}[1]{\marginpar{\begin{singlespace}\tiny #1
		\end{singlespace}}}
\addtolength{\marginparsep}{-0.25in}
\addtolength{\marginparsep}{4.6ex}
\addtolength{\marginparwidth}{-2ex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bell basis macros

% \def\Bzp{|^0_+\>}	% 00 + 11
% \def\Bop{|^1_+\>}	% 01 + 10
% \def\Bzm{|^0_-\>}	% 00 - 11
% \def\Bom{|^1_-\>}	% 01 - 10

%\def\Bzp{|\stackrel{\small 0}{\small +}\>}	% 00 + 11

%\def\Bzp{\left| \vphantom{x}^{\hspace*{1.25pt}0}_+ \right\>}	% 00 + 11
%\def\Bop{\left| \vphantom{x}^{\hspace*{1.25pt}1}_+ \right\>}	% 01 + 10
%\def\Bzm{\left| \vphantom{x}^{\hspace*{1.25pt}0}_- \right\>}	% 00 - 11
%\def\Bom{\left| \vphantom{x}^{\hspace*{1.25pt}1}_- \right\>}	% 01 - 10

\def\Bzp{| \beta_{00} \>}	% 00 + 11
\def\Bop{| \beta_{01} \>}	% 01 + 10
\def\Bzm{| \beta_{10} \>}	% 00 - 11
\def\Bom{| \beta_{11} \>}	% 01 - 10

\def\Bzpb{\< \beta_{00} |}	% 00 + 11
\def\Bopb{\< \beta_{01} |}	% 01 + 10
\def\Bzmb{\< \beta_{10} |}	% 00 - 11
\def\Bomb{\< \beta_{11} |}	% 01 - 10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 09-Sep-98 ILC: paragraph with double lines on left

\newcommand\mylfbox[1]{\noindent
       \ifCUP{\def\pwidth{4.5in}}{\def\pwidth{5.5in}}
		\begin{singlespace}\lfbox{~~\parbox[t]{\pwidth}{#1}}\\
			\end{singlespace}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 12-Nov-99 ILC: operator dagger macros

\def\adag{\udag{a}}
\def\bdag{\udag{b}}
\def\zdag{\udag{z}}
\def\cdag{\udag{c}}
\def\Adag{\udag{A}}
\def\Zdag{\udag{Z}}
\def\Bdag{\udag{B}}
\def\Xdag{\udag{X}}

% 14-Nov-99 ILC for implement.tex
\def\Ub{\Upsilon^{{\rm blue}}}
\def\deg{^\circ}

\newcommand{\vecfour}[4]{\left[\begin{array}{c}#1\\#2\\#3\\#4
			\end{array}\right]}

\def\Ybar{\bar Y}
\def\Xbar{\bar X}
\def\Rbar{\bar R}

\def\rand{$\not\! c$}	% for qcrypto random bit notation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \onpage{X}:
% 
% If the label X appears on this page, print ``on this page''.  If the label X
% appears on some other page, print ``on page~\pageref{X}''.

\newcommand{\onpage}[1]{%
	\label{onpage:#1}
	\typeout{ONPAGE: \pageref{onpage:#1}}
	\ifthenelse{\equal{{\pageref{#1}}}{{\pageref{onpage:#1}}}}%
	{\typeout{ONPAGE: #1 is on this page, \pageref{onpage:#1}}%
         on this page}%
	{\typeout{ONPAGE: #1 is on page \pageref{#1} (currently
	\pageref{onpage:#1}, thepage = \thepage)}%
         on page~\pageref{#1}}%
}

% \newcommand{\onpage}[1]{%
% \ifthenelse{\equal{{\pageref{#1}}}{{\thepage}}}%
% 	{\typeout{ONPAGE: #1 is on this page, \thepage}%
%          on this page}%
% 	{\typeout{ONPAGE: #1 is on page \pageref{#1}}%
%          on page~\pageref{#1}}%
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem environment (for PS)

\newenvironment{problem}{\begin{enumerate}%
		\renewcommand{\makelabel}[1]{%
  		    {\bf \TheType\refstepcounter{mycnt}\themycnt:~\hfill}%
		    \ifthenelse{\equal{##1}{}}{}{%
    		({\bf ##1})}%
			}}%
                        {\end{enumerate}}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% eqnarray environment without extra spaces around &=&

\newenvironment{myeqnarray}
	{\setlength{\arraycolsep}{1.5pt}%
	 \begin{eqnarray}}
	{\end{eqnarray}}

\renewcommand{\bea}{\begin{myeqnarray}}
\renewcommand{\eea}{\end{myeqnarray}}
\renewcommand{\beqn}{\begin{myeqnarray}}
\renewcommand{\eeqn}{\end{myeqnarray}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs (must go at end!) 11apr00 ILC

\usepackage{makeidx}
\input{mypsfig2.sty}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% upright \pi needs to use Euler fonts 16apr00 ILC

\DeclareSymbolFont{myletters}{U}{eur}{m}{n}
\DeclareMathSymbol\mymu     \mathord{myletters}{"016}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new html environment for references

\DeclareFixedFont{\mytt}{T1}{phv}{mc}{it}{9}
\renewenvironment{rawhtml}{\mytt}{}

%\ifCUP{\DeclareFixedFont{\mytt}{T1}{phv}{mc}{it}{9}}
%      {\def\mytt{\tt}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% font declarations for \arXiv 

\def\arXiv{\sl}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 27-May-00 new itemize environment for CUP

\newenvironment{myitemize}{\begin{itemize}}{\end{itemize}}

\ifCUP{
\renewenvironment{myitemize}{%
	\begin{flushleft}%
	\begin{list}{$\bullet$}{%
		\setlength{\leftmargin}{4mm}%
		\labelwidth \leftmargin%
		\addtolength{\labelwidth}{-\labelsep}%
		}%
	}%
	{\end{list}\end{flushleft}}
}{}
