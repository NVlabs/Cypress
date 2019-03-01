##
# @file   place_io.py
# @author Yibo Lin
# @date   Aug 2018
#

from torch.autograd import Function

import place_io_cpp

class PlaceIOFunction(Function):
    @staticmethod
    def forward(params):
        args = "gpf -config NORMAL"
        if "aux_file" in params.__dict__:
            args += " --bookshelf_aux_input %s" % (params.aux_file)
        if "lef_input" in params.__dict__:
            for lef in params.lef_input: 
                args += " --lef_input %s" % (lef)
        if "def_input" in params.__dict__:
            args += " --def_input %s" % (params.def_input)
        if "verilog_input" in params.__dict__:
            args += " --verilog_input %s" % (params.verilog_input)

        return place_io_cpp.forward(args.split(' '))
