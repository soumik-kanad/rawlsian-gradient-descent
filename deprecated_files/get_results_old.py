import os
import re

dir='experiments3'
outfile_race='results_race_'+dir+'.csv'
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
        #extracting eo

        # if '   Y     sex  R   EO' in content: 
        #     #handling spacing issue when the values are 0.0
        #     #bad way to handle but okay for now
        #     #print(len(content.split('   Y     sex  R   EO\n0  1  Female  1  ')))
        #     eo1,rest=content.split('   Y     sex  R   EO\n0  1  Female  1  ')[1].split('\n',1)
        #     eo2,rest=rest.split('1  1    Male  1  ')[1].split('\n',1)
        #     outfileline+=eo1+','+eo2+','
        #     #extracting pp
        #     pp1,rest=content.split('Y     sex  R   PP\n0  1  Female  1  ')[1].split('\n',1)
        #     pp2,rest=rest.split('1  1    Male  1  ')[1].split('\n',1)
        #     outfileline+=pp1+','+pp2+','
        #     #extracting dp
        #     dp1,rest=content.split('        A  R   DP\n0  Female  1  ')[1].split('\n',1)
        #     dp2,rest=rest.split('1    Male  1  ')[1].split('\n',1)
        #     outfileline+=dp1+','+dp2+','
        # if '   Y     sex  R        EO\n0  1  Female  1  ' in content:
        if 'sex' in content:
            # eo1,rest=content.split('   Y     sex  R        EO\n0  1  Female  1  ')[1].split('\n',1)
            eo1,rest=re.compile('   Y     sex  R[ \t]+EO\n0  1  Female  1  ').split(content)[1].split('\n',1)
            eo2,rest=rest.split('1  1    Male  1  ')[1].split('\n',1)
            outfileline+=eo1+','+eo2+','
            #extracting pp
            #pp1,rest=content.split('Y     sex  R        PP\n0  1  Female  1  ')[1].split('\n',1)
            pp1,rest=re.compile('Y     sex  R[ \t]+PP\n0  1  Female  1  ').split(content)[1].split('\n',1)
            pp2,rest=rest.split('1  1    Male  1  ')[1].split('\n',1)
            outfileline+=pp1+','+pp2+','
            #extracting dp
            # dp1,rest=content.split('        A  R        DP\n0  Female  1  ')[1].split('\n',1)
            dp1,rest=re.compile('        A  R[ \t]+DP\n0  Female  1  ').split(content)[1].split('\n',1)
            dp2,rest=rest.split('1    Male  1  ')[1].split('\n',1)
            outfileline+=dp1+','+dp2+','
            
            outfileline+='\n'
            with open(outfile_sex,'a') as f:
                f.write(outfileline)
        # elif '   Y                race  R        EO' in content:
        elif 'race' in content:
            #eo1,rest=content.split('   Y                race  R        EO\n0  1               White  1  ')[1].split('\n',1)
            eo1,rest=re.compile('   Y                race  R[ \t]+EO\n0  1               White  1  ').split(content)[1].split('\n',1)
            eo2,rest=rest.split('1  1  Asian-Pac-Islander  1  ')[1].split('\n',1)
            eo3,rest=rest.split('2  1  Amer-Indian-Eskimo  1  ')[1].split('\n',1)
            eo4,rest=rest.split('3  1               Other  1  ')[1].split('\n',1)
            eo5,rest=rest.split('4  1               Black  1  ')[1].split('\n',1)
            outfileline+=eo1+','+eo2+','+eo3+','+eo4+','+eo5+','
            #extracting pp
            #pp1,rest=content.split('   Y                race  R        PP\n0  1               White  1  ')[1].split('\n',1)
            pp1,rest=re.compile('   Y                race  R[ \t]+PP\n0  1               White  1  ').split(content)[1].split('\n',1)
            pp2,rest=rest.split('1  1  Asian-Pac-Islander  1  ')[1].split('\n',1)
            pp3,rest=rest.split('2  1  Amer-Indian-Eskimo  1  ')[1].split('\n',1)
            pp4,rest=rest.split('3  1               Other  1  ')[1].split('\n',1)
            pp5,rest=rest.split('4  1               Black  1  ')[1].split('\n',1)
            outfileline+=pp1+','+pp2+','+pp3+','+pp4+','+pp5+','
            #extracting dp
            #dp1,rest=content.split('                    A  R        DP\n0               White  1  ')[1].split('\n',1)
            dp1,rest=re.compile('                    A  R[ \t]+DP\n0               White  1  ').split(content)[1].split('\n',1)
            dp2,rest=rest.split('1  Asian-Pac-Islander  1  ')[1].split('\n',1)
            dp3,rest=rest.split('2  Amer-Indian-Eskimo  1  ')[1].split('\n',1)
            dp4,rest=rest.split('3               Other  1  ')[1].split('\n',1)
            dp5,rest=rest.split('4               Black  1  ')[1].split('\n',1)
            outfileline+=dp1+','+dp2+','+dp3+','+dp4+','+dp5+','


            outfileline+='\n'
            with open(outfile_race,'a') as f:
                f.write(outfileline)
