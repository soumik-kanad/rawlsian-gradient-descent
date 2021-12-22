import os
import re
import fire

def get_results(
    dir='experiments',
    outfile_race=None,
    outfile_sex=None,
):
    '''
    Accumulates the pandas tables printed in the output files 
    of experiments into csv files
    '''
    if not outfile_race:
        outfile_race='results_race_'+dir+'.csv'
    if not outfile_sex:
        outfile_sex='results_sex_'+dir+'.csv'

    exps=os.listdir(dir)
    exps.sort()
    for exp in exps:
        filename=os.path.join(dir,exp,'out.txt')
        if os.path.exists(filename):
            print(exp)
            with open(filename,'r') as f:
                content=f.read()
            outfileline=exp+","

            #extracting the accuracy
            accuracy=content.split('Test Accuracy: ')[1].split('\n')[0]
            outfileline+=accuracy+","
        
            if 'sex' in content:
                # eo1,rest=content.split('   Y     sex  R        EO\n0  1  Female  1  ')[1].split('\n',1)
                eo1,rest=re.compile('[ \t]+Y[ \t]+sex[ \t]+R[ \t]+EO\n0[ \t]+1[ \t]+Female[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                eo2,rest=re.split('1[ \t]+1[ \t]+Male[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=eo1+','+eo2+','
                #extracting pp
                #pp1,rest=content.split('Y     sex  R        PP\n0  1  Female  1  ')[1].split('\n',1)
                pp1,rest=re.compile('Y[ \t]+sex[ \t]+R[ \t]+PP\n0[ \t]+1[ \t]+Female[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                pp2,rest=re.split('1[ \t]+1[ \t]+Male[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=pp1+','+pp2+','
                #extracting dp
                # dp1,rest=content.split('        A  R        DP\n0  Female  1  ')[1].split('\n',1)
                dp1,rest=re.compile('[ \t]+A[ \t]+R[ \t]+DP\n0[ \t]+Female[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                dp2,rest=re.split('1[ \t]+Male[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=dp1+','+dp2+','
                
                outfileline+='\n'
                with open(outfile_sex,'a') as f:
                    f.write(outfileline)
            # elif '   Y                race  R        EO' in content:
            elif 'race' in content:
                #eo1,rest=content.split('   Y                race  R        EO\n0  1               White  1  ')[1].split('\n',1)
                eo1,rest=re.compile('Y[ \t]+race[ \t]+R[ \t]+EO\n0[ \t]+1[ \t]+White[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                # print(len(rest.split('1[ \t]+1[ \t]+Asian-Pac-Islander[ \t]+1[ \t]+')))
                # print(rest.split('1[ \t]+1[ \t]+Asian-Pac-Islander[ \t]+1[ \t]+'))
                eo2,rest=re.split('1[ \t]+1[ \t]+Asian-Pac-Islander[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                eo3,rest=re.split('2[ \t]+1[ \t]+Amer-Indian-Eskimo[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                eo4,rest=re.split('3[ \t]+1[ \t]+Other[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                eo5,rest=re.split('4[ \t]+1[ \t]+Black[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=eo1+','+eo2+','+eo3+','+eo4+','+eo5+','
                #extracting pp
                #pp1,rest=content.split('   Y                race  R        PP\n0  1               White  1  ')[1].split('\n',1)
                pp1,rest=re.compile('Y[ \t]+race[ \t]+R[ \t]+PP\n0[ \t]+1[ \t]+White[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                pp2,rest=re.split('1[ \t]+1[ \t]+Asian-Pac-Islander[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                pp3,rest=re.split('2[ \t]+1[ \t]+Amer-Indian-Eskimo[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                pp4,rest=re.split('3[ \t]+1[ \t]+Other[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                pp5,rest=re.split('4[ \t]+1[ \t]+Black[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=pp1+','+pp2+','+pp3+','+pp4+','+pp5+','
                #extracting dp
                #dp1,rest=content.split('                    A  R        DP\n0               White  1  ')[1].split('\n',1)
                dp1,rest=re.compile('A[ \t]+R[ \t]+DP\n0[ \t]+White[ \t]+1[ \t]+').split(content)[1].split('\n',1)
                dp2,rest=re.split('1[ \t]+Asian-Pac-Islander[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                dp3,rest=re.split('2[ \t]+Amer-Indian-Eskimo[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                dp4,rest=re.split('3[ \t]+Other[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                dp5,rest=re.split('4[ \t]+Black[ \t]+1[ \t]+',rest,1)[1].split('\n',1)
                outfileline+=dp1+','+dp2+','+dp3+','+dp4+','+dp5+','


                outfileline+='\n'
                with open(outfile_race,'a') as f:
                    f.write(outfileline)


if __name__=="__main__":
    fire.Fire(get_results)
