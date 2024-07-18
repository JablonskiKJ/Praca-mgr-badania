import pandas as pd
import numpy as np

# import time
# import progressbar
 
# widgets = [' [',
#          progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]
 
# bar = progressbar.ProgressBar(max_value=200,
#                               widgets=widgets).start()
 
# for i in range(200):
#     time.sleep(0.1)
#     bar.update(i)



class Rules:
    dfr: pd.DataFrame
    rules: pd.DataFrame
    unique_rules: pd.DataFrame
    split: pd.DataFrame
    ranking: pd.DataFrame
    
    def __init__(self, df):
        # Decision in last column
        self.df=df
        
    def oneBin(self): 
        self.df = self.df.loc[:, self.df.nunique() > 1]
        return self.df
        
    def makeDfr(self) -> pd.DataFrame:
        #init
        dfr=self.df.copy()
        range_len_df = range(len(dfr))
        # Creates DataFrame with lists containing row numbers with different values
        for col in dfr.columns:
            uniq = dfr[col].unique()
            dictionary=dict()
            for element in uniq:
                rn = np.array([row for row in range_len_df if element != dfr[col][row]])
                dictionary[element]=rn
            dfr[col] = dfr[col].map(dictionary)
        
        # Creates a DF with lists of row indexes that differ in column and have a different decision.
        cell = lambda i,col : np.array([x for x in dfr[col][i] if x in dfr[dfr.columns[-1]][i]])
        column = lambda col : np.array([cell(i,col) for i in range(len(dfr))], dtype=object)
        for col in dfr.columns[0:-1]:
            if any([len(z)!=0 for z in dfr[col]]):
                # bar = progressbar.ProgressBar(max_value=len(dfr), widgets=widgets).start()
                # bari=0
                # column=[]
                # for i in range(len(dfr)):
                #     column.append(cell(i,col))
                #     bar.update(bari)
                #     bari+=1
                # display(column)
                # dfr[col]=column(col)
                dfr[str(col)]=column(col)
                #dfr = dfr.assign(**{col: column(col)})
            
        self.dfr=dfr
        return dfr
    
    def makeRule(self, rowIndex: int, disp: bool) -> list:
        """Return the descriptors and percentages of the rows they separate."""
        if disp: print("row: "+str(rowIndex))
        row=self.dfr.loc[rowIndex].copy()
        if disp: print(row)
        lenna=len(row[-1]) #pełna ilość wierszy o różnych decyzjach
        dfcolumns=self.df.columns
        rule=[]
        split=[]
        
        
        actual=list(row[-1])
        last=[]
        
        #while (any(row)):
        while (any([len(z)!=0 for z in row])):
            if(actual==last):
                break
            last=list(row[-1])
            # index
            list_of_max_length = [len(row[col]) for col in row.index[0:-1]]
            index_of_max = list_of_max_length.index(max(list_of_max_length))
            
            index_name = dfcolumns[index_of_max] 
            if disp: print(index_name)
        
            
            conditional = row[index_name]
            lennb=len(conditional) #ilość odseparowanych wierszy
            percent=lennb/lenna #procent odseparowanych wierszy
            
            # usuwanie indexów odseparowanych wierszy
            for i in range(len(row)):
                #row[i] = [x for x in row[i] if x not in conditional]
                row[i] = np.array([x for x in row[i] if x not in conditional])
            if disp: print(row)

            # dodawanie do reguły                    # There is rowIndex
            rule.append([index_name, self.df[index_name][rowIndex]]) 
            split.append([index_name, percent])
            
            actual=list(row[-1])
        #else:
        if disp: print(rule)
        if disp: print(split)
        return rule, split
    
    def makeRules(self,disp=False) -> pd.DataFrame:
        """Return pd.DataFrame 'rules' with columns 'rule' and 'decision'"""
        range_len_df = range(len(self.dfr))
        rules_split = [self.makeRule(i,disp) for i in range_len_df]
        rules, split= zip(*rules_split)
        rules = pd.DataFrame({'rule': rules, 'split': split})
        rules = rules.assign(decision = self.df[self.df.columns[-1]])
        self.rules=rules
        return rules
    
    def makeSentence(self, rowIndex: int) -> str:
        """Returns a notation of a single rule"""
        #XXX: czy reguły muszą być sortowane? sorted_rule
        rule = self.rules['rule'][rowIndex]
        decision = self.rules['decision'][rowIndex]
        sentence='IF '
        for element in rule:
            sentence+=str(element[0]) + '=' + str(element[1]) + ' AND '
        sentence = sentence[:-5]
        sentence += ' THEN ' + str(decision)
        
        return sentence
    
    def makeSentences(self) -> pd.DataFrame:
        """Return pd.DataFrame 'rules' with column 'sentence' containing the rules as a string"""
        range_len_df = range(len(self.rules))
        sentence = [self.makeSentence(i) for i in range_len_df]
        
        self.rules = self.rules.assign(sentence = sentence)
        return self.rules
    
    #unikatowe reguły
    def uniqueRules(self) -> pd.DataFrame:
        self.unique_rules=self.rules.drop_duplicates(subset=['sentence'])
        return self.unique_rules
    
    #długość reguł
    def calculateLength(self) -> pd.DataFrame:
        pd.options.mode.chained_assignment = None
        self.unique_rules['length_of_rule']=[len(x) for x in self.unique_rules['rule']]
        pd.options.mode.chained_assignment = 'warn'
        return self.unique_rules
    
    #wsparcie
    def calculateSupport(self) -> pd.DataFrame:
        df2=self.unique_rules[['rule','decision']]
        z=[]
        for rule, dec in zip(df2['rule'], df2['decision']):
            n=self.df.copy()
            for arg, val in rule:
                n=n.where(n[arg]==val).dropna()
            n=n.where(n[n.columns[-1]]==dec).dropna()
            z.append(len(n))
        pd.options.mode.chained_assignment = None
        self.unique_rules['support']=z
        pd.options.mode.chained_assignment = 'warn'
        return self.unique_rules

    #pomyłki
    def calculateMisclassification(self) -> pd.DataFrame:
        df2=self.unique_rules[['rule','decision']]
        z=[]
        for rule, dec in zip(df2['rule'], df2['decision']):
            n=self.df.copy()
            for arg, val in rule:
                n=n.where(n[arg]==val).dropna()
            n=n.where(n[n.columns[-1]]!=dec).dropna()
            z.append(len(n))
        pd.options.mode.chained_assignment = None
        self.unique_rules['misclassification']=z
        pd.options.mode.chained_assignment = 'warn'
        return self.unique_rules
    
    #procent odseparowanych wierszy
    def separated_rows(self) -> pd.DataFrame:
        """Return pd.DataFrame with a percentage of separated rows"""
        df2=self.unique_rules.copy()
        #dodanie kolumn
        for col in self.df.columns[:-1]:
            df2[col]=0
        #dodanie procentu odseparowanych wierszy w kolumnach
        for rowi in df2.index:
            #print(rowi)
            for x in df2['split'][rowi]:
                df2.loc[rowi,x[0]]=x[1]
                #print(x)
    
        u=list(self.df.columns[:-1])
        u.append('support')
        u.append('length_of_rule')
        u=pd.Index(u)
        self.split=df2.loc[:,u]
        return self.split
    
    # version 3
    # 
    #ranking odseparowanych
    def makeRanking(self) -> pd.DataFrame:
        """Return pd.DataFrame 'ranking' of features"""
        ranking={}
       
        supported_rules = lambda split,name: len(split.loc[split[name]>0,name])
        rule_support = lambda split,name: max(split.loc[(split[name]==max(split[name])),'support']) if (split[name].any()) else None

        for x in self.split.columns[:-2]:
            ranking.update({x: (self.split[x].max(),supported_rules(self.split,x),rule_support(self.split,x))})


        # Sortowanie danych malejąco według kolumny 'criterium' i innych warunków
        ranking=pd.DataFrame(ranking).T
        #ranking[3]=ranking[0]*ranking[2]
        ranking=ranking.dropna()
        #ranking=ranking.sort_values([3,0],ascending=False)
        
        # version 2
        ranking=ranking.sort_values([1,0,2],ascending=False)
        #version 1
        # ranking=ranking.sort_values([0,1,2],ascending=False)
        
        self.ranking = ranking
        self.ranking.columns=['separation_rate','supported_rules','rule_support']                         
        return self.ranking