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

# generate random Gaussian values
from random import seed
from random import gauss
from random import randint

with open('test/fixed_random_gauss.txt', 'r') as f:
    LST = [[float(num) for num in line.split(',')] for line in f]
mini = min(LST)
maxi = max(LST)

LST = [x[0]+abs(mini[0]) for x in LST]

print(min(LST))
print(max(LST))

# seed random number generator
#seed(randint(0, 10000))
#mean=3E-6
#sigma=1E-6
precision = 9 # decimal points

clk_period = 100E-9
iterations = 50
simtime= iterations*clk_period # Time in milliseconds
#FLT_BIG_TIME=1E-12
FLT_SMALL_TIME=clk_period/2
FLT_BIG_TIME= FLT_SMALL_TIME

list_len=int(simtime/FLT_SMALL_TIME)
LST_VOLTAGE=[LST[0]*1E-6,0]*iterations
itr = 1
while itr in range(iterations):
    LST_VOLTAGE[2*itr] = LST[itr]*1E-6
    #print (''+ str(itr) +' File: ' + str(LST[itr]) + ' List: ' + str(LST_VOLTAGE[2*itr]))
    itr = itr + 1
# generate some Gaussian values
#for idx in range(list_len):
#    LST_VOLTAGE[idx] = round(gauss(mean, sigma),precision)
#    #print(LST_VOLTAGE[idx])

####### The variables below this line can be modified by the user ###########

# The user needs to modify this. This will be created or overwritten (in case the file exist)
TARGET_FILE=r'test/pwl_z.txt'

# The user needs to modify this to create an array of  voltages
# LST_VOLTAGE=[1.8 0]*15
# print (LST_VOLTAGE)
# This is used to fine tune the timing of the square wave


rise_fall_time = 1E-9


# uncomment the True to display the File Contents
BOOL_DISPLAY_FILE_CONTENTS=False  # | True









#####################################################
############ DO NOT MODIFY ANYTHING BELOW ###########
#####################################################
import time

starttime=time.time()

tmp='0 0\n' # create temporary storage of string so that we can display the contents laterz
#tmp=''

currTime = 0
ctr = 0
for x in LST_VOLTAGE:
#for x in range(iterations):
    tmp+='%.12f \t %.10f\n' % (currTime+rise_fall_time,x)
    #currTime+=FLT_SMALL_TIME
    print(ctr)
    #ctr +=1
    if(ctr%2):
        currTime+=FLT_BIG_TIME
    else:
        currTime+=FLT_SMALL_TIME
    tmp+='%.12f \t %.10f\n' % (currTime,x)
    ctr += 1

with open(TARGET_FILE,'w') as f:
    f.write(tmp)

if(BOOL_DISPLAY_FILE_CONTENTS):
    print ('========= File Contents: ==========\n')
    print (tmp)
    print ('========= EOF ==========\n\n')

print ("Process Done!")
print ("File Created: '%s'"  % (TARGET_FILE))

endtime=time.time()

elapsedtime=endtime-starttime

print ("Elapsed time in seconds:", elapsedtime)

# actually no need to do this. :)
LST_VOLTAGE=None
tmp=None


# EOF
