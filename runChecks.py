#!/usr/bin/python

"""
Replacement for test-math-dependencies target in Makefile.

Call script with '-h' as an option to see a helpful message.
"""

from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import os.path
import platform
import subprocess
import sys
import time
import glob
import re

winsfx = ".exe"
testsfx = "_test.cpp"
errors = []

def grep_patterns(type, folder, patterns_and_messages):
    """Checks the files in the provided folder for patterns 
    and if found returns an array of the provided message of
    the provided type with the line on which they occured.
    This check ignores comments.
    """
    folder.replace("/", os.sep)
    files = glob.glob(folder+os.sep+"**/*.hpp", recursive=True)
    for f in files:
        line_num = 0
        for line in open(f, "r"):
            line_num += 1
            # exclude comments
            # exclude line starting with "/*" or " * " and
            # if the matched patterns are behind "//"
            for p in patterns_and_messages:
                if not re.search("^ \* |^/\*", line) and not re.search(".*//.*"+p[0], line) and re.search(p[0], line):
                        errors.append(f + " at line " + str(line_num) + ":\n\t" + "[" + type + "] " + p[1])
def main():
    prim_checks = [
        ['<stan/math/rev/', 'File includes a stan/math/rev header file.'],
        ['stan::math::var', 'File uses stan::math::var.'],
        ['<stan/math/fwd/', 'File includes a stan/math/fwd header file.'],
        ['stan::math::fvar', 'File uses stan::math::fvar.'],
        ['<stan/math/mix/', 'File includes a stan/math/mix header file.']
    ]
    grep_patterns('prim', 'stan/math/prim', prim_checks )

    rev_checks = [
        ['<stan/math/fwd/', 'File includes a stan/math/fwd header file.'],
        ['stan::math::fvar', 'File uses stan::math::fvar.'],
        ['<stan/math/mix/', 'File includes a stan/math/mix header file.']
    ]
    grep_patterns('rev', 'stan/math/rev', rev_checks)

    scal_checks = [
        ['<stan/math/.*/arr/', 'File includes an array header file.'],
        ['<vector>', 'File includes an std::vector header.'],
        ['std::vector', 'File uses std::vector.'], 
        ['<stan/math/.*/mat/', 'File includes a matrix header file.'], 
        ['<Eigen', 'File includes an Eigen header.'],
        ['Eigen::', 'File uses Eigen.']
    ]
    grep_patterns('scal', 'stan/math/*/scal', scal_checks)

    arr_checks = [
        ['<stan/math/.*/mat/', 'File includes an matrix header file.'],
        ['<Eigen', 'File includes an Eigen header.'],
        ['Eigen::', 'File uses Eigen.']
    ]    
    grep_patterns('arr', 'stan/math/*/arr', arr_checks)

    cpp14_checks = [
        ['boost::is_unsigned<', 'File uses boost::is_unsigned instead of std::is_unsigned.'],
        ['<boost/type_traits/is_unsigned>',\
             'File includes <boost/type_traits/is_unsigned.hpp> instead of <type_traits>.'],
        ['boost::is_arithmetic<', 'File uses boost::is_arithmetic instead of std::is_arithmetic.'],
        ['<boost/type_traits/is_arithmetic.hpp>',\
             'File includes <boost/type_traits/is_unsigned.hpp> instead of <type_traits>.'],
        ['boost::is_convertible<', 'File uses boost::is_convertible instead of std::is_convertible.'],
        ['<boost/type_traits/is_convertible.hpp>',\
             'File includes <boost/type_traits/is_convertible.hpp> instead of <type_traits>.'],
        ['boost::is_same<', 'File uses boost::is_same instead of std::is_same.'],
        ['<boost/type_traits/is_same.hpp>',\
             'File includes <boost/type_traits/is_same.hpp> instead of <type_traits>.'],
        ['boost::enable_if_c<', 'File uses boost::enable_if_c instead of std::enable_if.'],
        ['boost::enable_if<', 'File uses boost::enable_if instead of std::enable_if.'],
        ['boost::disable_if<', 'File uses boost::disable_if instead of std::enable_if.'],
        ['<boost/utility/enable_if.hpp>',\
             'Replace \<boost/utility/enable_if.hpp\> with \<type_traits\>.']
    ]    
    grep_patterns('C++14', 'stan/math', cpp14_checks)

    if(len(errors) > 0):
        for e in errors:
            print(e)
        sys.exit(1)
        
if __name__ == "__main__":
    main()
