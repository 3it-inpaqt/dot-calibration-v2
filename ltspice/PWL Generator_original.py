"""  ---------------------------------------------------------------------------
Name:       LTSpice PWL.py
Purpose:    This tool is used generate PWL File for LTSpice.
            Originally, this generator is used to easily create
            a square wave signal.
            
            As of the moment, I am still looking for a good reference guide
            for PWL file so that I can create a flexible tool to generate 
            PWL file.
            
            For now, square wave is only supported
Author:     fwswdev @ github
Platform:
            Python 2.7.5
            No external libraries used
Created:    2013-10-16 14:20
Version:
            00.07.00
Copyright:
Licence:
  This is free and unencumbered software released into the public domain.
  
  Anyone is free to copy, modify, publish, use, compile, sell, or
  distribute this software, either in source code form or as a compiled
  binary, for any purpose, commercial or non-commercial, and by any
  means.
  
  In jurisdictions that recognize copyright laws, the author or authors
  of this software dedicate any and all copyright interest in the
  software to the public domain. We make this dedication for the benefit
  of the public at large and to the detriment of our heirs and
  successors. We intend this dedication to be an overt act of
  relinquishment in perpetuity of all present and future rights to this
  software under copyright law.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.
  
  For more information, please refer to <http://unlicense.org/>
--------------------------------------------------------------------------- """

####### The variables below this line can be modified by the user ###########

# The user needs to modify this. This will be created or overwritten (in case the file exist)
TARGET_FILE=r'test/pwl_original.txt'

# The user needs to modify this to create an array of  voltages
LST_VOLTAGE=[4e3,0]*5
#print LST_VOLTAGE
# This is used to fine tune the timing of the square wave
FLT_BIG_TIME=.1
FLT_SMALL_TIME=.2


# uncomment the True to display the File Contents
BOOL_DISPLAY_FILE_CONTENTS=False  # | True









#####################################################
############ DO NOT MODIFY ANYTHING BELOW ###########
#####################################################
import time

starttime=time.time()

tmp='0 0\n' # create temporary storage of string so that we can display the contents later

currTime = 0
ctr = 0
for x in LST_VOLTAGE:
    tmp+='%.10f \t %f\n' % (currTime+.0000000001,x)

    if(ctr%2):
        currTime+=FLT_BIG_TIME
    else:
        currTime+=FLT_SMALL_TIME
    tmp+='%.10f \t %f\n' % (currTime,x)
    ctr += 1

with open(TARGET_FILE,'w') as f:
    f.write(tmp)

if(BOOL_DISPLAY_FILE_CONTENTS):
    print('========= File Contents: ==========\n')
    print( tmp)
    print('========= EOF ==========\n\n')

print ("Process Done!")
print  ("File Created: '%s'"  % (TARGET_FILE))

endtime=time.time()

elapsedtime=endtime-starttime

print( "Elapsed time in seconds:", elapsedtime)

# actually no need to do this. :)
LST_VOLTAGE=None
tmp=None


# EOF
