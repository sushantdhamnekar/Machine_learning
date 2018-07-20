import random
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__== "__main__" :
    def parse(x):

        digit = loads(x)
        array = np.fromstring(b64decode(digit["data"]), dtype=np.ubyte)
        array = array.astype(np.float64)

        return (digit["label"], array)

    # read in the digits file. Digits is a list of 60,000 tuples,
    # each containing a labelled digit and its vector representation.
    with open("digits.base64.json", "r") as f:
        digits = list(map(parse, f.readlines()))

    #print("list1[0]: ", digits[1])

        #x = pd.DataFrame(list(digits(lab, vote)), columns=['annual_income', 'outlier'])

    ratio = int(len(list(digits)) * 0.25)
    #print(ratio)
    #print(digits[0])

    validation = digits[:ratio]
    training = digits[ratio:]



    array_0 = []
    array_1 = []


    for element in digits :
         array_0.append(element[0])
         array_1.append(element[1])



    #print('Number:',array_0[8])
    #print('Array:',array_1[8])
    #print('array_shape',len(array_1))
    D = pd.DataFrame(array_1)
    D.info()
    D.shape

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(D)
    #print('Print clusters:', kmeans.cluster_centers_)
    #print('size of cluster centers:', len(kmeans.cluster_centers_))
    clusters = kmeans.labels_
    #print('K_labels: ', clusters)
    #print('size of cluster:', len(clusters))
    D_clusters = pd.DataFrame({'A': clusters, 'B': array_0})
    #print('Coloumn info:' , D_clusters.columns)
    #print('Coloumn length:', len(D_clusters))

    #D_clusters.rename(columns = {'':'cluster_Nr'}, inplace=True)
    D_clusters.info()
    #D_clusters.describe(10)
    Group_clusters = D_clusters.groupby('A', axis=0)

    #print('Cluster numbers:', len(Group_clusters))
    #print('Cluster info::', Group_clusters.groups)



    group_ = []

    for name, group in Group_clusters:
        #print(name)
        #print(group)
        group_.append(group)

    #Group 0

    #print('print group 0:', group_[0])
    #print('print group 0:', group_[0]['A'])
    #print('print group 0:', group_[0]['B'])
    group_zero = pd.DataFrame({'A':group_[0]['A'],'B':group_[0]['B']})
    grp_0 = group_zero['B'].value_counts()
    print('group_zero data type:', grp_0.keys()[0])
    grp_0_value = grp_0.keys()[0]
    grp_0_array = np.array(group_zero['B'])
    #Group evaluation
    Range = len(grp_0_array) - 1
    num_0 = 0
    num_1 = 0

    for g in range(Range):
        if grp_0_array[g] == grp_0_value:
            num_1 = num_1 + 1
        else:
            num_0 = num_0 + 1
    print('Right clusters out of grp_0:', (num_1/Range)*100)
    print('wrong clusters percentage of grp_0:', (num_0/Range)*100)

    # Group 1


    group_one = pd.DataFrame({'A': group_[1]['A'], 'B': group_[1]['B']})
    grp_1 = group_one['B'].value_counts()
    print('group_one data type:', grp_1.keys()[0])
    grp_1_value = grp_1.keys()[0]
    grp_1_array = np.array(group_one['B'])
    # Group evaluation
    Range_1 = len(grp_1_array) - 1
    num_0_1 = 0
    num_1_1 = 0

    for g_1 in range(Range_1):
        if grp_1_array[g_1] == grp_1_value:
            num_1_1 = num_1_1 + 1
        else:
            num_0_1 = num_0_1 + 1
    print('Right clusters out of grp_1:', (num_1_1 / Range_1) * 100)
    print('wrong clusters percentage of grp_1:', (num_0_1 / Range_1) * 100)


    # Group 2

    group_two = pd.DataFrame({'A': group_[2]['A'], 'B': group_[2]['B']})
    grp_2 = group_two['B'].value_counts()
    print('group_two data type:', grp_2.keys()[0])
    grp_2_value = grp_2.keys()[0]
    grp_2_array = np.array(group_two['B'])
    # Group evaluation
    Range_2 = len(grp_2_array) - 1
    num_0_2 = 0
    num_1_2 = 0

    for g_2 in range(Range_2):
        if grp_2_array[g_2] == grp_2_value:
            num_1_2 = num_1_2 + 1
        else:
            num_0_2 = num_0_2 + 1
    print('Right clusters out of grp_2:', (num_1_2 / Range_2) * 100)
    print('wrong clusters percentage of grp_2:', (num_0_2 / Range_2) * 100)

 # Group 3

    group_three = pd.DataFrame({'A': group_[3]['A'], 'B': group_[3]['B']})
    grp_3 = group_three['B'].value_counts()
    print('group_three data type:', grp_3.keys()[0])
    grp_3_value = grp_3.keys()[0]
    grp_3_array = np.array(group_three['B'])
    # Group evaluation
    Range_3 = len(grp_3_array) - 1
    num_0_3 = 0
    num_1_3 = 0

    for g_3 in range(Range_3):
        if grp_3_array[g_3] == grp_3_value:
            num_1_3 = num_1_3 + 1
        else:
            num_0_3 = num_0_3 + 1
    print('Right clusters out of grp_3:', (num_1_3 / Range_3) * 100)
    print('wrong clusters percentage of grp_3:', (num_0_3 / Range_3) * 100)

    # Group 4

    group_four = pd.DataFrame({'A': group_[4]['A'], 'B': group_[4]['B']})
    grp_4 = group_four['B'].value_counts()
    print('group_four data type:', grp_4.keys()[0])
    grp_4_value = grp_4.keys()[0]
    grp_4_array = np.array(group_four['B'])
    # Group evaluation
    Range_4 = len(grp_4_array) - 1
    num_0_4 = 0
    num_1_4 = 0

    for g_4 in range(Range_4):
        if grp_4_array[g_4] == grp_4_value:
            num_1_4 = num_1_4 + 1
        else:
            num_0_4 = num_0_4 + 1
    print('Right clusters out of grp_4:', (num_1_4 / Range_4) * 100)
    print('wrong clusters percentage of grp_4:', (num_0_4 / Range_4) * 100)

    # Group 5

    group_five = pd.DataFrame({'A': group_[5]['A'], 'B': group_[5]['B']})
    grp_5 = group_five['B'].value_counts()
    print('group_five data type:', grp_5.keys()[0])
    grp_5_value = grp_5.keys()[0]
    grp_5_array = np.array(group_five['B'])
    # Group evaluation
    Range_5 = len(grp_5_array) - 1
    num_0_5 = 0
    num_1_5 = 0

    for g_5 in range(Range_5):
        if grp_5_array[g_5] == grp_5_value:
            num_1_5 = num_1_5 + 1
        else:
            num_0_5 = num_0_5 + 1
    print('Right clusters out of grp_5:', (num_1_5 / Range_5) * 100)
    print('wrong clusters percentage of grp_5:', (num_0_5 / Range_5) * 100)

    # Group 6

    group_six = pd.DataFrame({'A': group_[6]['A'], 'B': group_[6]['B']})
    grp_6 = group_six['B'].value_counts()
    print('group_six data type:', grp_6.keys()[0])
    grp_6_value = grp_6.keys()[0]
    grp_6_array = np.array(group_six['B'])
    # Group evaluation
    Range_6 = len(grp_6_array) - 1
    num_0_6 = 0
    num_1_6 = 0

    for g_6 in range(Range_6):
        if grp_6_array[g_6] == grp_6_value:
            num_1_6 = num_1_6 + 1
        else:
            num_0_6 = num_0_6 + 1
    print('Right clusters out of grp_6:', (num_1_6 / Range_6) * 100)
    print('wrong clusters percentage of grp_6:', (num_0_6 / Range_6) * 100)

    # Group 7

    group_seven = pd.DataFrame({'A': group_[7]['A'], 'B': group_[7]['B']})
    grp_7 = group_seven['B'].value_counts()
    print('group_seven data type:', grp_7.keys()[0])
    grp_7_value = grp_7.keys()[0]
    grp_7_array = np.array(group_seven['B'])
    # Group evaluation
    Range_7 = len(grp_7_array) - 1
    num_0_7 = 0
    num_1_7 = 0

    for g_7 in range(Range_7):
        if grp_7_array[g_7] == grp_7_value:
            num_1_7 = num_1_7 + 1
        else:
            num_0_7 = num_0_7 + 1
    print('Right clusters out of grp_7:', (num_1_7 / Range_7) * 100)
    print('wrong clusters percentage of grp_7:', (num_0_7 / Range_7) * 100)

    # Group 8

    group_eight = pd.DataFrame({'A': group_[8]['A'], 'B': group_[8]['B']})
    grp_8 = group_eight['B'].value_counts()
    print('group_eight data type:', grp_8.keys()[0])
    grp_8_value = grp_8.keys()[0]
    grp_8_array = np.array(group_eight['B'])
    # Group evaluation
    Range_8 = len(grp_8_array) - 1
    num_0_8 = 0
    num_1_8 = 0

    for g_8 in range(Range_8):
        if grp_8_array[g_8] == grp_8_value:
            num_1_8 = num_1_8 + 1
        else:
            num_0_8 = num_0_8 + 1
    print('Right clusters out of grp_8:', (num_1_8 / Range_8) * 100)
    print('wrong clusters percentage of grp_8:', (num_0_8 / Range_8) * 100)

    # Group 9

    group_nine = pd.DataFrame({'A': group_[9]['A'], 'B': group_[9]['B']})
    grp_9 = group_nine['B'].value_counts()
    print('group_nine data type:', grp_9.keys()[0])
    grp_9_value = grp_9.keys()[0]
    grp_9_array = np.array(group_nine['B'])
    # Group evaluation
    Range_9 = len(grp_9_array) - 1
    num_0_9 = 0
    num_1_9 = 0

    for g_9 in range(Range_9):
        if grp_9_array[g_9] == grp_9_value:
            num_1_9 = num_1_9 + 1
        else:
            num_0_9 = num_0_9 + 1
    print('Right clusters out of grp_9:', (num_1_9 / Range_9) * 100)
    print('wrong clusters percentage of grp_9:', (num_0_9 / Range_9) * 100)

    #Total calculation
    R_number = num_1 + num_1_1 + num_1_2 + num_1_3 + num_1_4 + num_1_5 + num_1_6 + num_1_7 + num_1_8 + num_1_9
    W_number = num_0 + num_0_1 + num_0_2 + num_0_3 + num_0_4 + num_0_5 + num_0_6 + num_0_7 + num_0_8 + num_0_9



    print('Right clusters out of  all groups:', (R_number / 60000) * 100)
    print('Wrong clusters percentage of all groups:', (W_number / 60000) * 100)