import pandas as pd

def equalised_odds(df, sensitive_arribute_col, sensitive_arribute_groups, target_col, prediction):
    """
    Calculates the equalised odds of a binary classification problem.
    :param df: pandas dataframe
    :sensitive_arribute_col: string, name of the column containing sensitive attribute
    :sensitive_arribute_groups: dictionary, names of the groups of sensitive attribute and their corresponding labels
    :param target_col: string, name of the target column
    :param prediction: predictions of the model
    :return: dataframe, equalised odds
    """
    # P(R=+|Y=y,A=a)=P(R=+|Y=y,A=b)\quad y\in \{+,-\}\quad \forall a,b\in A
    target = df[target_col].values
    sensitive_attribute = df[sensitive_arribute_col].values

    # print(len(prediction),len(target),len(sensitive_attribute))
    df1=pd.DataFrame({'R':prediction,'Y':target,'A':sensitive_attribute})
    print(sum(prediction))
    
    EO=[]
    R=[]
    Y=[]
    A=[]
    for label in [1, 0]:
        for group in sensitive_arribute_groups:
            group_label = sensitive_arribute_groups[group]
        
            numerator=len(df1[(df1['R']==1) & (df1['Y']==label) & (df1['A']==group_label)])
            denominator=len(df1[(df1['Y']==label) & (df1['A']==group_label)])
            # print("Y=",label,",A=",group,", EO=",numerator,"/",denominator)
            Y.append(label)
            A.append(group)
            R.append(1)
            EO.append(numerator/(denominator+1e-12))

    equalised_odds = {'Y':Y,sensitive_arribute_col:A,'R':R,'EO':EO}
    # print(equalised_odds)
    return pd.DataFrame(equalised_odds)


def predictive_parity(df,sensitive_arribute_col, sensitive_arribute_groups, target_col, prediction):
    # P(Y=+|R=+,A=a)=P(Y=+|R=+,A=b)\quad \forall a,b\in A

    target = df[target_col].values
    sensitive_attribute = df[sensitive_arribute_col].values

    df1=pd.DataFrame({'R':prediction,'Y':target,'A':sensitive_attribute})

    PP=[]
    R=[]
    Y=[]
    A=[]
    for label in [1, 0]:
        for group in sensitive_arribute_groups:
            group_label = sensitive_arribute_groups[group]

            numerator=len(df1[(df1['Y']==label) &(df1['R']==1) & (df1['A']==group_label)])
            denominator=len(df1[(df1['R']==1) & (df1['A']==group_label)])

            Y.append(label)
            A.append(group)
            R.append(1)
            PP.append(numerator/(denominator+1e-12))
    
    predictive_parity = {'Y':Y,sensitive_arribute_col:A,'R':R,'PP':PP}
    return pd.DataFrame(predictive_parity)

def demographic_parity(df,sensitive_arribute_col, sensitive_arribute_groups, prediction):
    # P(R=+|A=a)=P(R=+|A=b)\quad \forall a,b\in A
    sensitive_attribute = df[sensitive_arribute_col].values

    df1=pd.DataFrame({'R':prediction,'A':sensitive_attribute})

    DP=[]
    R=[]
    A=[]
    for group in sensitive_arribute_groups:
        group_label = sensitive_arribute_groups[group]

        numerator=len(df1[(df1['R']==1) & (df1['A']==group_label)])
        denominator=len(df1[(df1['A']==group_label)])

        A.append(group)
        R.append(1)
        DP.append(numerator/(denominator+1e-12))
    
    demographiv_parity = {'A':A,'R':R,'DP':DP}
    return pd.DataFrame(demographiv_parity)

    
