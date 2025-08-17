# coding: utf-8

import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, metrics
import urllib
import pickle

class WAF(object):

    def __init__(self):
        good_query_list = self.get_query_list('goodqueries3.txt')
        bad_query_list = self.get_query_list('bad_queries3.txt')
        
        good_y = [0 for i in range(0,len(good_query_list))]
        bad_y = [1 for i in range(0,len(bad_query_list))]

        queries = bad_query_list + good_query_list
        y = bad_y + good_y
        ''''
        #converting data to vectors  
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)

        
        X = self.vectorizer.fit_transform(queries)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

        self.lgs = LogisticRegression()
        self.lgs.fit(X_train, y_train)
        self.dtc = tree.DecisionTreeClassifier()
        self.dtc.fit(X_train, y_train)
        self.linear_svm=LinearSVC(C=1)
        self.linear_svm.fit(X_train, y_train)
        #self.rfc = RandomForestClassifier(n_estimators=200)
        #self.rfc.fit(X_train, y_train)
        
        #y_pred = self.lgs.predict(X_test)
        #score_test = metrics.accuracy_score(y_test, y_pred)
        #print(score_test)

        #print('Model accuracy:{}'.format(self.lgs.score(X_test, y_test)))
        # Save the trained logistic regression model
        with open('lgs3.joblib', "wb") as file:
            pickle.dump(self.lgs, file)
        with open('dtc3.joblib', "wb") as file:
            pickle.dump(self.dtc, file)
        with open('linear_svm_model3.joblib', "wb") as file:
            pickle.dump(self.linear_svm, file)
        with open('random_forest_model2.joblib', "wb") as file:
            pickle.dump(self.rfc,file)

        # Save the fitted TfidfVectorizer
        with open('vectorizer3.pickle', 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)
        '''
        
    
    '''def predict(self,new_queries):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        models = {
            'Logistic': self.lgs,
            'DecisionTree': self.dtc,
            'LinearSVM': self.linear_svm,
            'RandomForest': self.rfc
        }

        res_list = []

        for model_name, model in models.items():
            start_time = time.time()
            res = model.predict(X_predict)
            detection_time = time.time() - start_time
            res_list.append({'model': model_name, 'res': res, 'detection_time': detection_time})

        for result in res_list:
            print(f"Model: {result['model']}, Result: {result['res']}, Detection time: {result['detection_time']}")
        return res_list
    '''
    '''def predict(self, new_queries, true_labels):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        models = {
            'Logistic': self.lgs,
            'DecisionTree': self.dtc,
            'LinearSVM': self.linear_svm,
            'RandomForest': self.rfc
        }

        res_list = []

        for model_name, model in models.items():
            start_time = time.time()
            res = model.predict(X_predict)
            detection_time = time.time() - start_time

        # Calculate accuracy
            accuracy = accuracy_score(true_labels, res)

            res_list.append({
                'model': model_name,
                'res': res,
                'detection_time': detection_time,
                'accuracy': accuracy
            })

        for result in res_list:
            print(f"Model: {result['model']}, Accuracy: {result['accuracy']:.4f}, "
                  f"Detection time: {result['detection_time']}")

        return res_list
    
        res1 = self.lgs.predict(X_predict)
        res2 = self.dtc.predict(X_predict)
        res3 = self.linear_svm.predict(X_predict)
        res4 = self.rfc.predict(X_predict)
        
        res_list.append({'model':'lgs','res':res1})
        res_list.append({'model':'dtc','res':res2})
        res_list.append({'model':'linear_svm','res':res3})
        res_list.append({'model':'rfc','res':res4})
        print(res_list)
        return res_list
        '''
    '''def predict(self, new_queries, true_labels=None):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        models = {
            'Logistic': self.lgs,
            'DecisionTree': self.dtc,
            'LinearSVM': self.linear_svm,
            'RandomForest': self.rfc
        }

        res_list = []

        for model_name, model in models.items():
            start_time = time.time()
            res = model.predict(X_predict)
            detection_time = time.time() - start_time
        
            metrics_dict = {
                'model': model_name,
                'predictions': res,
                'runtime': detection_time,
            }
        
            if true_labels is not None:
                accuracy = accuracy_score(true_labels, res)
                precision = precision_score(true_labels, res)
                metrics_dict.update({
                    'accuracy': accuracy,
                    'precision': precision
                })
        
            res_list.append(metrics_dict)

        # Print results
        for result in res_list:
            if true_labels is not None:
                print(f"Model: {result['model']}")
                print(f"Accuracy: {result['accuracy']:.4f}")
                print(f"Precision: {result['precision']:.4f}")
                print(f"Runtime: {result['runtime']:.6f} seconds")
                print("-" * 30)
            else:
                print(f"Model: {result['model']}")
                print(f"Predictions: {result['predictions']}")
                print(f"Runtime: {result['runtime']:.6f} seconds")
                print("-" * 30)
    
        return res_list    '''
    def _print_evaluation_report(self, results, has_labels):
        """Print formatted evaluation report"""
        print("\n" + "="*80)
        print("QUERY PREDICTION REPORT".center(80))
        print("="*80)
    
        # Print per-query results
        print(f"\nAnalyzed {len(results['queries'])} queries")
    
        for i, query_result in enumerate(results['per_query']):
            print(f"\nQuery {i+1}: {query_result['query'][:100]}{'...' if len(query_result['query']) > 100 else ''}")
        
            if has_labels:
                print(f"True Label: {'BAD' if query_result['true_label'] else 'GOOD'}")
        
            for model_name, pred in query_result['predictions'].items():
                print(f"  {model_name}:")
                print(f"    Prediction: {'BAD' if pred['prediction'] else 'GOOD'}")
                if has_labels:
                    print(f"    Correct: {'✓' if pred['correct'] else '✗'}")
                print(f"    Runtime: {pred['runtime']:.6f}s")
    
        # Print aggregate metrics if labels available
        if has_labels:
            print("\n" + "="*80)
            print("MODEL PERFORMANCE METRICS".center(80))
            print("="*80)
        
            # Print confusion matrices
            print("\nConfusion Matrices:")
            for model_name, metrics in results['aggregate'].items():
                print(f"\n{model_name}:")
                print(f"  True Positives: {metrics['true_positives']}")
                print(f"  False Positives: {metrics['false_positives']}")
                print(f"  True Negatives: {metrics['true_negatives']}")
                print(f"  False Negatives: {metrics['false_negatives']}")
        
            # Print metrics table
            headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Avg Runtime", "Total Runtime"]
            print("\n{:<20} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}".format(*headers))
        
            for model_name, metrics in results['aggregate'].items():
                print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<12.6f} {:<12.6f}".format(
                    model_name,
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    metrics['avg_runtime'],
                    metrics['total_runtime']
                ))
    
        print("="*80 + "\n")

    def predict_from_file(self, file_path, true_labels=None, delimiter='\t'):
        # Read queries from file
        queries = []
        file_labels = []
    
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(delimiter)
                queries.append(urllib.parse.unquote(parts[0]))
                if len(parts) > 1:
                    file_labels.append(int(parts[1]))
    
        if not queries:
            raise ValueError("No valid queries found in test file")
    
        # Determine which labels to use
        if true_labels is None:
            if file_labels:
                true_labels = file_labels
                print("Using labels from file")
            else:
                print("No labels provided - running in prediction-only mode")
        elif len(true_labels) != len(queries):
            raise ValueError(f"Number of true labels ({len(true_labels)}) doesn't match queries ({len(queries)})")
    
        # Vectorize queries
        X_predict = self.vectorizer.transform(queries)
    
        # Prepare models
        models = {
            'LogisticRegression': self.lgs,
            'DecisionTree': self.dtc,
            'LinearSVM': self.linear_svm
            #'RandomForest': self.rfc
        }
    
        results = {
            'per_query': [],
            'aggregate': {},
            'queries': queries,
            'true_labels': true_labels if true_labels is not None else None
        }
    
        # Process each model
        for model_name, model in models.items():
            model_results = {
                'predictions': [],
                'runtimes': [],
                'correct': 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }
        
            # Predict each query individually
            for i in range(len(queries)):
                start_time = time.perf_counter()
                pred = model.predict(X_predict[i:i+1])[0]  # Predict single query
                runtime = time.perf_counter() - start_time
            
                model_results['predictions'].append(pred)
                model_results['runtimes'].append(runtime)
            
                if true_labels is not None:
                    true = true_labels[i]
                    model_results['correct'] += int(pred == true)
                
                    # Update confusion matrix counts
                    if pred == 1 and true == 1:
                        model_results['true_positives'] += 1
                    elif pred == 1 and true == 0:
                        model_results['false_positives'] += 1
                    elif pred == 0 and true == 0:
                        model_results['true_negatives'] += 1
                    else:
                        model_results['false_negatives'] += 1
        
            # Store model results
            results['aggregate'][model_name] = model_results
        
            # Calculate metrics if labels available
            if true_labels is not None:
                tp = model_results['true_positives']
                fp = model_results['false_positives']
                tn = model_results['true_negatives']
                fn = model_results['false_negatives']
            
                model_results['accuracy'] = (tp + tn) / len(queries)
                model_results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                model_results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                model_results['f1'] = 2 * (model_results['precision'] * model_results['recall']) / \
                                     (model_results['precision'] + model_results['recall']) \
                                     if (model_results['precision'] + model_results['recall']) > 0 else 0
                model_results['avg_runtime'] = np.mean(model_results['runtimes'])
                model_results['total_runtime'] = np.sum(model_results['runtimes'])
    
        # Prepare per-query results
        for i in range(len(queries)):
            query_result = {
                'query': queries[i],
                'predictions': {}
            }
        
            if true_labels is not None:
                query_result['true_label'] = true_labels[i]
        
            for model_name in models:
                query_result['predictions'][model_name] = {
                    'prediction': results['aggregate'][model_name]['predictions'][i],
                    'runtime': results['aggregate'][model_name]['runtimes'][i]
                }
            
                if true_labels is not None:
                    query_result['predictions'][model_name]['correct'] = \
                        (results['aggregate'][model_name]['predictions'][i] == true_labels[i])
        
            results['per_query'].append(query_result)
    
        # Print comprehensive report
        self._print_evaluation_report(results, true_labels is not None)
    
        return results

    def get_query_list(self,filename):
        '''directory = str(os.getcwd())
        # directory = str(os.getcwd())+'/module/waf'
        filepath = directory + "/" + filename
            data = open(filepath,'r').readlines()
            query_list = []
            for d in data:
                d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
                # print(d)
                query_list.append(d)
            return list(set(query_list))'''
        directory = str(os.getcwd())
        filepath = directory + "/" + filename
        with open(filepath, 'r', encoding='utf-8') as file:  # specify encoding as 'utf-8'
            data = file.readlines()
        query_list = []
        for d in data:
            d = str(urllib.parse.unquote(d))  # converting url encoded data to simple string
            query_list.append(d)
        return list(set(query_list))


    #tokenizer function, this will make 3 grams of each query
    def get_ngrams(self,query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0,len(tempQuery)-3):
            ngrams.append(tempQuery[i:i+3])
        return ngrams

if __name__ == '__main__':
    w = WAF()
    #with open('lgs2.pickle','wb') as output:
        #pickle.dump(w,output)

    #with open('lgs.pickle','rb') as input:
        #w = pickle.load(input)

    # Load the saved vectorizer
    with open('vectorizer3.pickle', 'rb') as vectorizer_file:
        w.vectorizer = pickle.load(vectorizer_file)

    # Load the saved models
    with open('lgs3.joblib', 'rb') as file:
        w.lgs = pickle.load(file)
    with open('dtc3.joblib', 'rb') as file:
        w.dtc = pickle.load(file)
    with open('linear_svm_model3.joblib', 'rb') as file:
        w.linear_svm = pickle.load(file)
    #with open('random_forest_model.joblib', 'rb') as file:
    #    w.rfc = pickle.load(file)    
    

    # X has 46 features per sample; expecting 7  youqude  cuowu  
    '''w.predict(['www.foo.com/id=1<script>alert(1)</script>','www.foo.com/name=admin\' or 1=1','abc.com/admin.php',
    '"><svg onload=confirm(1)>','test/q=<a href="javascript:confirm(1)>','q=../etc/passwd'], [1, 1, 0, 1, 1, 1])
    w.predict(['www.test.com/?x=<script>alert(1)</script>','www.test.com/admin.php?username=admin\' or 1=1','http://www.test.com/?file=../../../etc/passwd','payload=<a href="javascript:alert(1)">','file=index.php','www.google.com/?search=hello'], [1, 1, 1, 1, 0, 0])'''
    '''true_safe_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    true_malicious_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'''
    true_file_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #w.predict_from_file('good.txt', true_safe_labels, delimiter='\t')
    #w.predict_from_file('bad1.txt', true_malicious_labels, delimiter='\t')
    w.predict_from_file('aa.txt', true_file_labels, delimiter='\t')