# based on xml.simp -> start_time and end_time -> split using sox

import os
import sys

# use .simp as the source for .wav file splitting

def wavfn(apath):
    wavdict = dict() # key=id, value=full.path of .wav
    for awavfn in os.listdir(apath):
        fullwavpath = os.path.join(apath, awavfn)    
        aid = awavfn.replace('.wav', '')
        wavdict[aid] = fullwavpath
    return wavdict

def xmlfn(apath):
    xmldict = dict() # key=id, value=full.path of .xml.simp
    for axmlfn in os.listdir(apath):
        if not axmlfn.endswith('.xml.simp'):
            continue
        axmlfn2 = os.path.join(apath, axmlfn)
        aid = axmlfn.replace('.xml.simp', '')
        print('obtain id: {}\t{}'.format(axmlfn, aid))
        xmldict[aid] = axmlfn2
    return xmldict

def bad(lines):
    # all lines with col.num <= 2
    for line in lines:
        alen = len(line.strip().split('\t'))
        if alen > 2:
            return False
    return True


def addXMLline(xmlfn, xmlbw, lines):
    if lines == None or len(lines) == 0 or bad(lines):
        return

    stime = lines[0].split('\t')[0]
    etime = lines[-1].split('\t')[1]

    adur = float(etime) - float(stime)
    print('accumulated duration={}'.format(adur))
    if adur >= 17.0:
        # a bug:
        print('a dur={}, xmlfn={}, stime={}, etime={}'.format(adur, xmlfn, stime, etime))
        #exit()

    # stime    etime    col1    col2    col3    col4    col5 
    cols = [ '' for _ in range(5) ]
    for line in lines:
        acols = line.split('\t')
        for i in range(2, len(acols)):
            cols[i-2] += ' ' + acols[i]

    newline = stime + '\t' + etime + '\t' + '\t'.join(cols)
    xmlbw.write(newline + '\n')

def proc1file(fullxmlfn):
    fullxmlfn_out = fullxmlfn + ".comb"
    with open(fullxmlfn) as xmlbr:
        with open(fullxmlfn_out, 'w') as xmlbw:
            # update: 2021.Nov.13 and Nov.17 -> combine too short audios into >= 3.0 seconds single-audio NOTE
            accumulated_dur = 0.0
            accumulated_stime = 0.0
            accumulated_lines = list() # for new text-output
            pre_stime, pre_etime = 0.0, 0.0 # need to consider the distance between "pre_etime" and "cur_stime"
            is_first = True

            for axmlline in xmlbr.readlines():
                # start.time end.time ortho plainortho phonetic
                axmlline = axmlline.strip()
                cols = axmlline.split('\t')
                stime, etime = float(cols[0]), float(cols[1])
                # NOTE a bug was here [2022 April 04] len(cols) =2 and this is a noise...with long duration...
                # when face a noise, skip it and restart (to ensure not to include this noise)
                is_noise = (len(cols) == 2) # cols.num = 2, 5, 7

                dur = etime - stime
                if is_first: # 前一个period是noise，或者本period是xml文件的开头:
                    accumulated_dur = dur
                    accumulated_stime = stime
                    accumulated_lines.append(axmlline)
                    pre_stime, pre_etime = stime, etime
                    is_first = False
                    #continue # -> if this one line is already >= 3 seconds, then we can write it to xml file!
                
                #partwavfn = '{}_{}_{}.wav'.format(fullwavfn, stime, etime)
                # 1. noise; 2. accumulated_dur >= 3.0 seconds; 3. current start.time - former end.time >= 1.0 seconds
                if is_noise or accumulated_dur >= 3.0 or dur >= 15.0 or (stime - pre_etime >= 1.0):
                    addXMLline(fullxmlfn, xmlbw, accumulated_lines)

                    # re-initialize the xml line array:
                    accumulated_lines = list()
                    if is_noise:
                        # skip current line, and restart as "first":
                        accumulated_dur = 0.0
                        accumulated_stime = 0.0
                        pre_stime, pre_etime = 0.0, 0.0
                        is_first = True
                    else:
                        accumulated_lines.append(axmlline)
                        # re-assign value to current two variables:
                        accumulated_stime = stime
                        accumulated_dur = dur
                        pre_stime, pre_etime = stime, etime
                else:
                    # not a noise and current accumulated_duration < 3 seconds
                    accumulated_dur += dur
                    accumulated_lines.append(axmlline)
                    pre_stime, pre_etime = stime, etime
                    # do not need to change "accumulated_stime"

                #res = os.system(acmd)
                #print(res, acmd)
            # after "for" loop, save the final audio file:
            addXMLline(fullxmlfn, xmlbw, accumulated_lines)

def procpath(apath):
    # apath = 'core' and 'noncore'
    #axmlpath = '/raid/xianchaow/asr/csj/XML/BaseXML/core' if apath=='core' else '/raid/xianchaow/asr/csj/XML/BaseXML/noncore'
    #axmlpath = '/workspace/asr/csj/XML/BaseXML/core' if apath=='core' else '/workspace/asr/csj/XML/BaseXML/noncore'
    axmlpath = apath
    xmldict = xmlfn(axmlpath)

    for aid in xmldict:
        fullxmlfn = xmldict[aid]
        proc1file(fullxmlfn)

#for apath in ['core', 'noncore']:
#    procpath(apath)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} <xml.path>".format(sys.argv[0]))
        exit(1)

    apath = sys.argv[1]
    procpath(apath)

