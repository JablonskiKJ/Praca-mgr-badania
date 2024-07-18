import pandas as pd
import numpy as np

class Classifier:
    ATRN: int
    
    #nazwa kolumny zawierającej reguły
    RULE_COL='rule'
    SUP_COL='support'
    
    DEC_COL: str
    
    test_df: pd.DataFrame
    test_decision: pd.Series
    rules: pd.DataFrame
    ranking: pd.DataFrame
    confusion_matrix: pd.DataFrame
    
    #Wczytanie zbioru testowego
    def loadTestSet(self, filepath: str) -> pd.DataFrame:
        '''Returns pd.DataFrame with a test set from a file with a specified path.'''
        self.test_df = pd.read_csv(filepath,dtype=object, sep=",")
        self.test_decision=self.test_df[self.test_df.columns[-1]].copy()
        return self.test_df

    def loadRuleSet(self, rulefilepath: str) -> pd.DataFrame:
        '''Returns pd.DataFrame with a set of rules from a file with the specified path'''
        f=open(rulefilepath,"r")
        
        #wczytanie nagłówka pliku z regułami
        f.readline()
        #ilość atrybutów
        atrn=int(f.readline().split(' ')[1])
        self.ATRN=atrn-1
        # print(atrn)

        #odczytanie atrybutów
        columns=[f.readline().split(' ')[1] for i in range(atrn)]
        
        #atrybut decyzyjny
        self.DEC_COL=columns[-1]
        # print(self.DEC_COL)

        #ilość klas decyzyjnych
        decn=int(f.readline().split(' ')[1])
        # print(decn)

        #!skip klas decyzyjnych
        for i in range(decn):
            f.readline()

        #ilość reguł
        rulesn=int(f.readline().split(' ')[1])
        # print(rulesn)

        #stworzenie pustego test_df dla reguł
        rules=pd.DataFrame(index=np.arange(rulesn), columns=columns)
        rules[self.RULE_COL] = np.empty((len(rules), 0)).tolist()

        #Wczytanie reguł
        for i in range(rulesn):
            #odczytanie linii
            #line=f.readline().rsplit()[0]
            #line=line.split("=>")
            line=f.readline().split()
            line,sup=line
            #wsparcie
            sup=int(sup)
            line=line.split("=>")
            #display(line)
            #reguła
            rule=[x[1:-1].split("=") for x in line[0].split("&")]
            #display(rule)
            decision=line[1].split("=")[1].split('[')[0]
            #display(decision)
            rules.at[i,self.DEC_COL]=decision
            rules.at[i,self.RULE_COL]= rule
            rules.at[i,self.SUP_COL]= sup
            
            for x in rule:
                rules.at[i,x[0]]=x[1]
        
        self.rules=rules
        return self.rules

    def loadRanking(self,rankingfilepath) -> pd.DataFrame:
        self.ranking = pd.read_csv(rankingfilepath, sep=",", index_col=0)
        return self.ranking

    # def classify(self) -> pd.DataFrame:
    #     rules=self.rules
    #     test_df=self.test_df
        
    #     test_df[self.DEC_COL]=np.nan
        
    #     # dla każdego obiektu
    #     for index, row in test_df.iterrows():
    #         # display(row)
    #         rules_copy=rules.copy()
    #         condition=pd.Series(True,index=rules_copy.index)
    #         #reguły pasujące dla danego atrybutu
    #         for attr, val in zip(row.index[:-1], row.to_numpy()):
    #             condition=condition&((rules_copy[attr]==val)|(rules_copy[attr].isnull()))
    #         # display(rules_copy)
            
    #         rules_copy=rules_copy[condition]
    #         test_df.at[index,self.DEC_COL]=rules_copy[[self.DEC_COL,self.SUP_COL]].groupby(self.DEC_COL).sum().idxmax()[self.SUP_COL] if len(rules_copy) else np.nan

    #     # #klasyfikacja - wersja z pierwszą pasującą regułą
    #     # test_df[self.DEC_COL]=np.nan
    #     # for i in rules.index:
    #     #     if(not any(pd.isna(test_df[self.DEC_COL]))):
    #     #         # print('Sklasyfikowano wszystkie obiekty')
    #     #         break
    #     #     #reguła
    #     #     rule=rules.loc[i,self.RULE_COL]
    #     #     #decyzja
    #     #     decision=rules.loc[i,self.DEC_COL]
    #     #     #zgodność poszczególnych deskryptorów 
    #     #     con=[test_df[x[0]]==x[1] for x in rule]
    #     #     #tylko do jeszcze nie przypisanych
    #     #     condition=pd.isna(test_df[self.DEC_COL])
    #     #     #stworzenie pełnego warunku
    #     #     for n in con:
    #     #         condition=condition&n

    #     #     #przypisanie decyzji do warunków
    #     #     test_df.loc[condition,self.DEC_COL]=decision
    #     # else:
    #     #     #przypisanie decyzji o najwyższym wsparciu do jeszcze nie sklasyfikowanych
    #     #     test_df.loc[pd.isna(test_df[self.DEC_COL]),self.DEC_COL] = rules[[self.DEC_COL,self.SUP_COL]].groupby(self.DEC_COL).apply(lambda x: sum(x[self.SUP_COL])).idxmax() if len(rules) else self.test_decision[0]
    #     # # else:
    #     # #     print('Koniec reguł..')
    
    #     #Macierz klasyfikacji
    #     un=self.test_decision.unique()
    #     un1= ['pred. '+x for x in un]
    #     confusion_matrix=pd.DataFrame(0,un1,un)

    #     #porównanie 
    #     for z in un:
    #         for y,y1 in zip(un,un1):
    #            confusion_matrix[z][y1]=len(self.test_decision.loc[lambda x:x==z].index
    #                .intersection(test_df[self.DEC_COL].loc[lambda x:x==y].index))

    #     #acc=sum([confusion_matrix[y][y1] for y,y1 in zip(un,un1)])/len(test_df)
    #     #print('dokładność klasyfikacji wynosi: '+f'{acc:.2f}')
    #     #display(confusion_matrix)
        
    #     self.confusion_matrix=confusion_matrix
    #     return self.confusion_matrix
    
    def classify(self):
        rules=self.rules
        test=self.test_df
        DEC_COL=self.DEC_COL
        SUP_COL=self.SUP_COL
        test[DEC_COL]=np.nan
        
        def classify_row(row, rules):
            mask=np.array([(rules[x]==row[x])|(rules[x].isnull()) for x in row.index[:-1]])
            mask2 = np.array([all(x) for x in zip(*mask)])
            rules_copy=rules.iloc[mask2.nonzero()]
            return rules_copy[[DEC_COL,SUP_COL]].groupby(DEC_COL).sum().idxmax()[SUP_COL] if len(rules_copy) else np.nan
        
        test[DEC_COL]=test.apply(lambda row: classify_row(row, rules), axis=1)
        acc=(test[DEC_COL]==self.test_decision).value_counts(True).get(True, 0.0)
        
        #Macierz klasyfikacji
        un=self.test_decision.unique()
        un1= ['pred. '+x for x in un]
        confusion_matrix=pd.DataFrame(0,un1,un)

        #porównanie - uzupełnienie macierzy
        for z in un:
            for y,y1 in zip(un,un1):
                confusion_matrix[z][y1]=len(self.test_decision.loc[lambda x:x==z].index
                    .intersection(test[self.DEC_COL].loc[lambda x:x==y].index))

        self.confusion_matrix=confusion_matrix
        return self.confusion_matrix, acc