import os
from time import time

from analyst.viewhandler import viewPreproc, view_common_predict, viewnormdist_daynum, view_all_sts_compare

import numpy as np

from config.configutil import getpath
from preproc.preprocessor import preproc, removenoiseprefile, countprefile
from proc.cnn_processor import cnn_train
from rtklib.solparser import pasredsol

if __name__ == '__main__':
    # sts = ['alic', 'bake', 'bjfs', 'meli', 'penc', 'picl', 'pova',  'sch2','nril', 'tixi', 'yakt', 'hkws']
    # sts = [ 'bake', 'bjfs', 'meli', 'penc', 'picl', 'pova',  'nril', 'tixi', 'yakt', 'hkws']
    sts = ['alic']
    ######
    # sts = [ 'alic','alic_time','alic_azi','alic_ele','alic_p_res','alic_cr_res','alic_snr','alic_valid_flag','alic_cycle','alic_fix','alic_lock','alic_out','alic_slip_cnt','alic_data_reject']
    # sts = [ 'bake','bake_time','bake_azi','bake_ele','bake_p_res','bake_cr_res','bake_snr','bake_valid','bake_cycle','bake_amb','bake_fix','bake_outage','bake_slip_count','bake_reject_count']
    # sts = ['bjfs', 'bjfs_time', 'bjfs_azi', 'bjfs_ele', 'bjfs_p_res', 'bjfs_cr_res', 'bjfs_snr', 'bjfs_valid', 'bjfs_cycle', 'bjfs_amb', 'bjfs_fix', 'bjfs_outage', 'bjfs_slip_count', 'bjfs_reject_count']
    # sts = ['meli', 'meli_time', 'meli_azi', 'meli_ele', 'meli_p_res', 'meli_cr_res', 'meli_snr', 'meli_valid','meli_cycle', 'meli_amb', 'meli_fix', 'meli_outage', 'meli_slip_count', 'meli_reject_count']
    # sts = ['penc', 'penc_time', 'penc_azi', 'penc_ele', 'penc_p_res', 'penc_cr_res', 'penc_snr', 'penc_valid','penc_cycle', 'penc_amb', 'penc_fix', 'penc_outage', 'penc_slip_count', 'penc_reject_count']
    # sts = ['picl', 'picl_time', 'picl_azi', 'picl_ele', 'picl_p_res', 'picl_cr_res', 'picl_snr', 'picl_valid','picl_cycle', 'picl_amb', 'picl_fix', 'picl_outage', 'picl_slip_count', 'picl_reject_count']
    # sts = ['pova', 'pova_time', 'pova_azi', 'pova_ele', 'pova_p_res', 'pova_cr_res', 'pova_snr', 'pova_valid','pova_cycle', 'pova_amb', 'pova_fix', 'pova_outage', 'pova_slip_count', 'pova_reject_count']
    # sts = ['nril', 'nril_time', 'nril_azi', 'nril_ele', 'nril_p_res', 'nril_cr_res', 'nril_snr', 'nril_valid','nril_cycle', 'nril_amb', 'nril_fix', 'nril_outage', 'nril_slip_count', 'nril_reject_count']
    # sts = ['tixi', 'tixi_time', 'tixi_azi', 'tixi_ele', 'tixi_p_res', 'tixi_cr_res', 'tixi_snr', 'tixi_valid','tixi_cycle', 'tixi_amb', 'tixi_fix', 'tixi_outage', 'tixi_slip_count', 'tixi_reject_count']
    # sts = ['yakt', 'yakt_time', 'yakt_azi', 'yakt_ele', 'yakt_p_res', 'yakt_cr_res', 'yakt_snr', 'yakt_valid','yakt_cycle', 'yakt_amb', 'yakt_fix', 'yakt_outage', 'yakt_slip_count', 'yakt_reject_count']
    # sts = [ 'hkws','hkws_time','hkws_azi','hkws_ele','hkws_p_res','hkws_cr_res','hkws_snr','hkws_valid','hkws_cycle','hkws_amb','hkws_fix','hkws_outage','hkws_slip_count','hkws_reject_count']
    # sts = [ 'sch2','sch2_time','sch2_azi','sch2_ele','sch2_p_res','sch2_cr_res','sch2_snr','sch2_valid','sch2_cycle','sch2_amb','sch2_fix','sch2_outage','sch2_slip_count','sch2_reject_count']

    # batch train
    # prenums =range(5, 6)
    # prenums =range(8, 9)
    # for st in sts:
    #     fnum = countprefile(getpath('preprocpath', st))
    #     for prenum in prenums:
    #          train_range = range(fnum - prenum, fnum - 2)
    #          tstart = time()
    #          print('begin:', st, fnum - prenum)
    #          # 3-1 cnn proc
    #          cnn_train(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range)
    #          print('end:', st, fnum - prenum, ' consume time:', (time() - tstart), 's')
    # batch test
    # prenums =range(10, 11)
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # for st in sts:
    #     fnum = countprefile(getpath('preprocpath', st))
    #     test_range = range(fnum - 2, fnum)
    #     for prenum in prenums:
    #         train_range = range(fnum - prenum, fnum - 2)
    #         normdist = cnn_predict(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range, test_range, True)
    #
    #     print(st, np.array(normdist))
    # batch get txt
    # prenums = range(3, 21)
    # for st in sts:
    #     cnnworkpath = getpath('cnnworkpath', st)
    #     cnnviewpath = getpath('cnnviewpath', st)
    #     preprocpath = getpath('preprocpath', st)
    #     fnum = countprefile(preprocpath)
    #     test_range = range(fnum - 2, fnum)
    #     pred_res = np.zeros((len(prenums), 7))
    #     for prenum in prenums:
    #         train_range = range(fnum - prenum, fnum - 2)
    #         # print(prenum-2, end=' ')
    #         normdist = cnn_predict(preprocpath, cnnworkpath, train_range, test_range, False)
    #         print(st,',', prenum-2, ',', np.array(normdist))
    #         pred_res[prenum - 3, 0] = prenum-2
    #         pred_res[prenum - 3, 1:8] = normdist
    #     predictname = st + '_'+str(prenums[0])+'_'+str(prenums[-1])
    #     np.savez(cnnviewpath + predictname, pred_res=pred_res)
    # for st in sts:
    #     cnnviewpath = getpath('cnnviewpath', st)
    #     predictname = st + '_'+str(prenums[0])+'_'+str(prenums[-1])
    #     pred_res = np.load(cnnviewpath + predictname+'.npz')['pred_res']
    #     viewnormdist_daynum(pred_res,predictname,cnnviewpath)
    # prenums =[10, 20]
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # first_pred_res = np.zeros((len(sts),6))
    # second_pred_res = np.zeros((len(sts),6))
    # i = 0
    # for st in sts:
    #     fnum = countprefile(getpath('preprocpath', st))
    #     test_range = range(fnum - 2, fnum)
    #     for prenum in prenums:
    #         train_range = range(fnum - prenum, fnum - 2)
    #         normdist = cnn_predict(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range, test_range, False)
    #         if prenum == 10:
    #             first_pred_res[i, 0:3] = normdist[0:3]
    #             second_pred_res[i, 0:3] = normdist[3:6]
    #         else:
    #             first_pred_res[i, 3:6] = normdist[0:3]
    #             second_pred_res[i, 3:6] = normdist[3:6]
    #     i = i+1
    # np.savez('all_sts_compare', first_pred_res=first_pred_res,second_pred_res= second_pred_res,sts = sts)
    # all_sts_compare = np.load('all_sts_compare.npz')
    # first_pred_res = all_sts_compare['first_pred_res']
    # second_pred_res = all_sts_compare['second_pred_res']
    # sts = all_sts_compare['sts']
    # view_all_sts_compare(first_pred_res, second_pred_res, sts)
    # 统计不同元素对结果的影响
    # stcount = len(sts)
    # for i in np.arange(stcount):
    #     st = sts[i]
    #     fnum = countprefile(getpath('preprocpath', st))
    #     print(st, fnum)
    #     train_range = range(fnum - 10, fnum - 2)
    #     test_range = range(fnum - 2, fnum)
        # cnn_train_mask(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range,i-1)
        # normdist = cnn_predict_mask(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range, test_range, i-1)
        # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        # print(st, np.array(normdist))

    for st in sts:
        # 1 parse
        # pasredsol(getpath('eachsolpath', st), getpath('parsedsolpath', st), getpath('parsedpospath', st),
        #           getpath('parsedstatpath', st))
        # 2-0 preproc
        # preproc(getpath('parsedsolpath', st),
        #         getpath('parsedstatpath', st), getpath('preprocpath', st), getpath('soltype', st))
        # 2-1 remove noise data
        # removenoiseprefile(getpath('preprocpath', st), getpath('soltype', st))
        # 3 proc
        tstart = time()
        fnum = countprefile(getpath('preprocpath', st))
        print(st, fnum)
        train_range = range(fnum - 10, fnum - 2)
        test_range = range(fnum - 2, fnum)
        # 3-1 cnn proc
        cnn_train(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range)
        # normdist = cnn_predict(getpath('preprocpath', st), getpath('cnnworkpath', st), train_range, test_range, True)
        # print(st, normdist)


        # 4-0 for latex
        # test_range = range(fnum - 2, fnum-1)
        # test_range = range(fnum - 1, fnum)
        # latextempl = loadTemplate('latex_tab_cnn_item.tpl')
        # for i in test_range:
        #     preName = str(train_range[0]) + '_' + str(train_range[-1]) + '_'+str(i)+'_'
        #     outcontent = outCnn4LatexTableItem(getpath('workspace', st), st, preName, latextempl)
        # print(outcontent)
        # 5-0 for train data view
        # view_range = range(fnum - 10, fnum)
        # viewPreproc(getpath('preprocpath', st), view_range)
        # 6-0 clear attention
        # clearWorkspace(getpath('workspace', st)
    