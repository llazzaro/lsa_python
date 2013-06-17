#!/usr/bin/python

import sys
from optparse import OptionParser


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--file', action='open file',
                type='string', dest='file', help='file top open')

    options, args = parser.parse_args()

    if not options.file:
        print("Filename is required!")
        parser.print_help()
        sys.exit(1)
