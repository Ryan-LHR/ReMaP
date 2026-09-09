# 返回的是我们的方法优化版的采样用例
import copy

from utils import save_point_print


def select_my_optimize(model, x_target, y_test, selectsize,
                       prob_vectors, num_classes, prioritized_dict, budget_size_lst):
    "Official implementation of MCP"
    # x = np.zeros((selectsize, 28, 28, 1))
    # y = np.zeros((selectsize,))

    # act_layers = model.predict(x_target)
    act_layers = prob_vectors

    # dicratio = [[] for i in range(100)]  # 只用90，闲置10个
    # dicindex = [[] for i in range(100)]
    dicratio = [[] for _ in range(num_classes ** 2)]  # 只用90，闲置10个
    dicindex = [[] for _ in range(num_classes ** 2)]

    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index

        # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
        # dicratio[max_index * 10 + sec_index].append(ratio)
        # dicindex[max_index * 10 + sec_index].append(i)

        pair_index = max_index * num_classes + sec_index
        dicratio[pair_index].append(ratio)
        dicindex[pair_index].append(i)

    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    # for i in range(selectsize):
    #     test = x_target[selected_lst[i]]
        # x[i] = x_target[selected_lst[i]]
        # y[i] = y_test[selected_lst[i]]

    for i, (budget, _) in enumerate(prioritized_dict.items()):
        dicratio_temp = copy.deepcopy(dicratio)
        dicindex_temp = copy.deepcopy(dicindex)
        budget_size = budget_size_lst[i]
        save_point_print(f'Selecting under budget size={budget_size}')
        selected_lst = select_from_firstsec_dic(budget_size, dicratio_temp, dicindex_temp)

        # result = selected_lst
        assert len(selected_lst) == budget_size, "Selected list size does not match budget."
        prioritized_dict[budget] = selected_lst

    return prioritized_dict
    # return x, y

# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex):
    selected_lst = []
    tmpsize = selectsize
    # tmpsize保存的是采样大小，全程都不会变化

    noempty = no_empty_number(dicratio)
    print(selectsize)
    print(noempty)
    # 待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    # while selectsize >= noempty:
    while selectsize >= noempty and noempty > 0 :
        # for i in range(100):
        for i in range(len(dicratio)):
            if len(dicratio[i]) != 0:  # 非空就选一个最大的出来
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                # if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)
        # print(selectsize)
        # print(noempty)
    # selectsize<noempty
    # no_empty_number(dicratio)
    print(selectsize)

    # 剩下少量样本没有采样，比如还存在30类别非空，但是只要采样10个，此时我们取30个最大值中的前10大
    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]  # 剩下多少就申请多少
        max_index_tmp = [0 for i in range(selectsize)]
        # for i in range(100):
        for i in range(len(dicratio)):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]  # 吧样本序列号存在此列表中
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
        # print(len(selected_lst))
    # print(selected_lst)
    assert len(selected_lst) == tmpsize, "Selected list size does not match budget."
    return selected_lst


def find_second(act):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(len(act)):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(len(act)):
        if i == max_index:
            continue
        if act[i] > second_max:  # 第2大加一个限制条件，那就是不能和max_一样
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio  # ratio是第二大输出达到最大输出的百分比

#配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty
