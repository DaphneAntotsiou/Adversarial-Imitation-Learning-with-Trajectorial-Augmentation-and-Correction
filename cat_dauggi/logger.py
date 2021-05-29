__author__ = 'DafniAntotsiou'

'''
This script is heavily based on @openai/baselines.gail.statistics
'''

from baselines.logger import *

class Logger(Logger):
    def dumpkvs(self, stdout=True):
        if self.level == DISABLED: return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter) and \
                    (stdout or (not stdout and not isinstance(fmt, HumanOutputFormat))):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()
