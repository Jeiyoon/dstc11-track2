from matplotlib import pyplot as plt

"""
ground-truth												                    22.0	22.0			69.0
multi-qa-MiniLM-L6-cos-v1		77.4	54.5	63.2	70.5	79.1	74.6	80.0	22.0	24.0	48.0	3449.0	71.9
all-MiniLM-L6-v2				74.7	52.8	61.2	70.5	74.9	72.7	83.4	22.0	25.0	47.0	3449.0	73.4
all-MiniLM-L12-v2				73.2	47.1	57.1	66.0	72.2	69.0	83.1	22.0	25.0	47.0	3449.0	73.4
all-mpnet-base-v2				72.8	41.6	60.2	65.6	76.3	70.6	86.5	22.0	26.0	42.0	3449.0	82.1
"""

# y = [26, 25, 25, 24] # k
y = [82.1, 73.4, 73.4, 71.9] # intent
x_gt_k = [22, 22, 22, 22]
x_nmi_value = [72.8, 73.2, 74.7, 77.4]
x_ari_value = [41.6, 47.1, 52.8, 54.5]
x_f1_value = [70.6, 69.0, 72.7, 74.6]
x_coverage_value = [86.5, 83.1, 83.4, 80.0]

# linestyle = '--', ':'
plt.xlabel('Example Coverage')
plt.ylabel('# of Utterances per Intent')
plt.plot(x_coverage_value, y, color='firebrick', marker='o')
# plt.plot(x_gt_k, y, color='peachpuff')
plt.axhline(y=69.0, color='peachpuff', linestyle='dashdot')
plt.vlines(x=86.5, ymin=69.0, ymax=82.1, color='gray', linestyle='--')
plt.vlines(x=83.1, ymin=69.0, ymax=73.4, color='gray', linestyle='--')
plt.vlines(x=83.4, ymin=69.0, ymax=73.4, color='gray', linestyle='--')
plt.vlines(x=80.0, ymin=69.0, ymax=71.9, color='gray', linestyle='--')
plt.legend(['Example Coverage According to # of Utterances per Intent', 'Max # of Utterances per Intent'])
plt.show()

# table
# score = '81.0	64.2	66.5	75.5	80.6	78.0	86.2	22.0	25.0'
# result = '&'
#
# for s in score.split("\t"):
#     print(s)
#     result += '  ' + s + '  &'
#
# print(result)


# github
# score = '62.5	37.0	50.5	67.6	51.7	58.6	97.9	22.0	36.0'
# result = '|'
#
# for s in score.split("\t"):
#     print(s)
#     result += '  ' + s + '  |'
#
# print(result)