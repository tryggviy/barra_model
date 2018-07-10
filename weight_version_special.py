

#XIAOTIAN 2018/03/29

#CNE5S_100_Asset_Exposure: THE ASSET EXPOSURE FOR EVERY STOCK
#CHN_LOCALID_Asset_ID: THE ASSET ID USED TO MATCH LOCAL ID AND BARRA ID
#CNE5_Daily_Asset_Price
#CNE5S_100_DlyFacRet
#CNE5_100_Asset_DlySpecRet
#JXHY1_GMY001_Position
#future_pos.csv

#After testing, it is found that the enddate in ID table means the expiray date for the stock
#Thus, we only examine those stocks which are currently active

from multiprocessing import Process, Manager, Lock
import datetime as dt
from pandas import DataFrame, concat, read_table, read_csv, merge, isnull, merge
import time
from copy import copy
from tushare import get_k_data
from math import sqrt
from time import strftime
from os import mkdir
from numpy import argmax, maximum, prod, matrix, dot, multiply, array, var, std
# multiprocessing.set_start_method('spawn')
class PassingError(Exception): 
        """Error encountered while passing arguments."""
        pass

# def trans_table(tgt_df,target_date):
#     tgt_df.columns = ['account','ticker','wind_code','volume','market_price','market_value']
#     tgt_df1 = tgt_df[tgt_df.wind_code != '204001.SH'].copy()
#     tgt_df1 = tgt_df1[tgt_df1.market_value.notnull()].copy()
#     weights_pre = list(tgt_df1.market_value/sum(tgt_df1.market_value))
#     tgt_df1['weight'] = weights_pre
#     tgt_df1['date'] = [target_date]*len(tgt_df1)
#     tgt_df2 = tgt_df1[['wind_code','ticker','date','weight']].copy()
#     tgt_df2.rename(columns={'wind_code':'symbol','ticker':'sedol'}, inplace = True)
#     return tgt_df2

def Del_Narows(tgt_df, col_name):
    nalist = list(tgt_df.index[isnull(tgt_df[col_name])].values)
    return tgt_df.drop(nalist)

def trans_table(tgt_df):
    tgt_df.columns = ['symbol','value','change']
    tgt_df2 = Del_Narows(tgt_df,'value')
    tgt_df2['value'] = tgt_df2.value.astype('float64')
    tgt_df2['weight'] = (tgt_df2.value/sum(tgt_df2.value))
    tgt_df2['symbol'] = tgt_df2.symbol.astype('str')
    tgt_df2['symbol'] = tgt_df2.symbol.apply(lambda x : ('0'*(6-len(x))+x) if len(x) < 6 else x)
    return tgt_df2[['symbol','weight']]

def idx_weight(df_list):
    length = len(df_list)

    for ia in range(length):
        df_list[ia].columns = ['Barrid','weight'+str(ia)]

    emydf = df_list[0]

    for ib in range(1,length):
        emydf = merge(emydf, df_list[ib], how='outer')

    emydf.fillna(0,inplace=True)
    weight_list = [None]*length

    for ic in range(length):
        weight_list[ic] = emydf['weight'+str(ic)]

    emydf['weight'] = sum(weight_list)

    return emydf[['Barrid','weight']]


def sumcols(df1, df2):
    final_wts = merge(df1, df2, how='outer')
    final_wts.fillna(0,inplace=True)
    final_wts['weight'] = final_wts.weight+final_wts.weight
    return final_wts['Barrid','weight']

def latest_id(local_id,iddf):
    dfa = iddf[iddf.AssetID.apply(lambda x: x == 'CN' + local_id[:6])]
    return dfa[dfa.EndDate == max(dfa.EndDate)].values[0,0]

#A function used to relate the localids and barraids
def lcl_bra(df1,df2):
    lcldict = DataFrame(columns=['Barrid', 'weight'])
    count = 0
    nanlist = []
    nanweight = []
    for index,row in df1.iterrows():
        try:
            latest_id(row[0],df2)
        except:
            print(row[0]+' This is not included in barra')
            nanlist.append(row[0])
            nanweight.append(row[1])
        else:
            lcldict = lcldict.append({'Barrid':latest_id(row[0],df2),'weight':row[1]},ignore_index=True)
        count += 1
    lcldict.weight = lcldict.weight/(1-sum(nanweight))
    return lcldict,nanlist

#a dict which contains the connection between localids and barraids

#Construct the Exposure Matrix
def make_expmat(nmwgts,expdf,fctlist):
    exposure_mat = []
    for pair in nmwgts:
        fct_rt = dict()
        fct_df = expdf[expdf.Barrid == pair[0]]
        for index,row in fct_df.iterrows():
            fct_rt[row[1]] = row[2]
        exposure_mat.append(fct_rt)
    return exposure_mat

#Convert exp_matrix to a data frame and replace all null values with 0
def make_fulldf(fctlist,expm):
    df_empty = DataFrame(columns=fctlist)
    for el in expm:
        df_empty = df_empty.append(el,ignore_index=True)
    return df_empty

#get the weights vector from wigts_pairs
def getwigt(alist):
    epylist = len(alist)*[None]
    for ii in range(len(alist)):
        epylist[ii] = alist[ii][1]
    return epylist


def getwigt_name(alist):
    epylist = len(alist)*[None]
    for ii in range(len(alist)):
        epylist[ii] = alist[ii][0]
    return epylist

def getwigt_localid(alist):
    epylist = len(alist)*[None]
    for ii in range(len(alist)):
        epylist[ii] = alist[ii][2]
    return epylist


def DftoPairlist(df):
    emptylist = [0]*len(df)
    for idx in range(len(df)):
        temp = [0,0]
        temp[0] = (df.iloc[idx])[0]
        temp[1] = (df.iloc[idx])[1]
        emptylist[idx] = temp
    return emptylist

def max_drawdown(timeseries):
    i = argmax(maximum.accumulate(timeseries) - timeseries)
    j = argmax(timeseries[:i])
    return (float(timeseries[i]) / timeseries[j]) - 1.

def acmultiply(aseries):
    nalist = []
    for i in range(len(aseries)):
        nalist.append(prod(aseries[:i+1]))
    return nalist

def date_cvt(astr):
    if len(astr) == 10:
        return astr[0:4]+astr[5:7]+astr[8:10]
    
    elif len(astr) == 8:
        return astr[0:4]+'-'+astr[4:6]+'-'+astr[6:8]
    
    else:
        raise PassingError('The passed argument is not as expected')

def mainfunc(global_lock, read_path_input, target_date_rt, target_wgt_dt, futures_pos, \
            asset_allocation, index_hs300p, index_zz500p,index_sz50p, index35_test, prod_name
             ,print_middle = False,print_result = True):

    CNE5S_100_Asset_Exposure = read_table(read_path_input+'CNE5S_100_Asset_Exposure.'+target_date_rt, sep = '|', header=2)

    CHN_LOCALID_Asset_ID = read_csv(read_path_input+'CHN_LOCALID_Asset_ID.'+target_date_rt, sep = '|', header=1)
    CNE5S_100_DlyFacRet = read_table(read_path_input+'CNE5S_100_DlyFacRet.'+target_date_rt, sep = '|', header=2)
    
    CNE5_100_Asset_DlySpecRet = read_table(read_path_input+'CNE5_100_Asset_DlySpecRet.'+target_date_rt, sep = '|', header=2)
    CNE5_100_Asset_DlySpecRet.rename(columns={'!Barrid':'Barrid'}, inplace = True)
    
    hs300,nanlisths = lcl_bra(read_csv(read_path_input+'hs300w_'+target_wgt_dt+'.csv'),CHN_LOCALID_Asset_ID)
    zz500,nanlistzz = lcl_bra(read_csv(read_path_input+'zz500w_'+target_wgt_dt+'.csv'),CHN_LOCALID_Asset_ID)
    sz50,nanlistsz = lcl_bra(read_csv(read_path_input+'sz50w_'+target_wgt_dt+'.csv'),CHN_LOCALID_Asset_ID)

    dt_fmt2 = (target_wgt_dt)[:4]+'-'+(target_wgt_dt)[4:6]+'-'+(target_wgt_dt)[6:8]

    if_price = index_hs300p[index_hs300p.date == dt_fmt2].close.values[0]
    ic_price = index_zz500p[index_zz500p.date == dt_fmt2].close.values[0]
    ih_price = index_sz50p[index_sz50p.date == dt_fmt2].close.values[0]

    if_pos = futures_pos[futures_pos.DATE == target_wgt_dt].IF.values[0]
    ic_pos = futures_pos[futures_pos.DATE == target_wgt_dt].IC.values[0]
    ih_pos = futures_pos[futures_pos.DATE == target_wgt_dt].IH.values[0]

    # target_allocation = asset_allocation[asset_allocation.DATES == dt_fmt2]
    # shares_ratio = target_allocation.SHARES.values[0]/target_allocation.TOTAL.values[0]
    # futures_ratio = target_allocation.FUTURES.values[0]/target_allocation.TOTAL.values[0]

    wgt_CH = trans_table(read_csv(read_path_input+prod_name+'_shares_'+target_wgt_dt+'.csv'))
    wgt_CH['weight'] = wgt_CH.weight

    if index35_test == 'p1f0':
        wgt_CH1,nanlist = lcl_bra(wgt_CH,CHN_LOCALID_Asset_ID)
        final_wights = wgt_CH1

    elif index35_test == 'p1f1':
        wgt_CH1,nanlist = lcl_bra(wgt_CH,CHN_LOCALID_Asset_ID)

        if ic_pos+if_pos+ih_pos != 0:
            ic_weight = ic_pos*ic_price*200
            if_weight = if_pos*if_price*300
            ih_weight = ih_pos*ih_price*300
            future_sum = ic_weight+if_weight+ih_weight

            hs300['weight'] = -(if_weight/future_sum) * hs300.weight * 0.01
            zz500['weight'] = -(ic_weight/future_sum) * zz500.weight * 0.01
            sz50['weight'] = -(ih_weight/future_sum) * sz50.weight * 0.01
            # aa = sum(sum(hs300.weight)+sum(zz500.weight)+sum(sz50.weight))
            input_wigt_list = [wgt_CH1, hs300, zz500, sz50]

            idxweights_table = idx_weight(input_wigt_list)

            final_wights = idxweights_table
        else:
            final_wights = wgt_CH1

        nanlist.extend(nanlisths)
        nanlist.extend(nanlistzz)
        nanlist.extend(nanlistsz)

    elif index35_test == 'p0f1':
        ic_weight = ic_pos*ic_price*200
        if_weight = if_pos*if_price*300
        ih_weight = ih_pos*ih_price*300
        future_sum = ic_weight+if_weight+ih_weight

        hs300['weight'] = -(if_weight/future_sum) * hs300.weight * 0.01
        zz500['weight'] = -(ic_weight/future_sum) * zz500.weight * 0.01
        sz50['weight'] = -(ih_weight/future_sum) * sz50.weight * 0.01

        input_wigt_list = [hs300, zz500, sz50]

        idxweights_table = idx_weight(input_wigt_list)
        final_wights = idxweights_table
        
        nanlist =[]
        nanlist.extend(nanlisths)
        nanlist.extend(nanlistzz)
        nanlist.extend(nanlistsz)
    else:
        raise PassingError('The passed argument is not as expected in mainfunc')

    final_wights_pair = DftoPairlist(final_wights)
    
    CNE5S_100_Asset_Exposure.rename(columns={'!Barrid':'Barrid'}, inplace = True)
    CHN_LOCALID_Asset_ID.rename(columns={'!Barrid':'Barrid'}, inplace = True)
    CNE5S_100_DlyFacRet.rename(columns={'!Factor':'Factor'}, inplace = True)
    CNE5S_100_FacRet1 = CNE5S_100_DlyFacRet[CNE5S_100_DlyFacRet.DataDate == int(target_date_rt)]
    
    # Show first lines of all tables read in in file

    #a dict which contains the connection between localids and barraids
    
    wigts_pairs = final_wights_pair.copy()
    dlytotal_specrt = 0
    for pair in wigts_pairs:
        spec_rt = float(CNE5_100_Asset_DlySpecRet[CNE5_100_Asset_DlySpecRet.Barrid == pair[0]].SpecificReturn.values[0])
        dlytotal_specrt += 0.01 * spec_rt * pair[1]
    #Amendation to the factor returns column
    #Get the factor list name
        
    CNE5S_100_FacRet_mat = matrix(list(CNE5S_100_FacRet1.DlyReturn)).T
    factor_list = list(CNE5S_100_FacRet1.Factor)
    exp_matrix = make_expmat(wigts_pairs,CNE5S_100_Asset_Exposure,factor_list)
    exposure_matrix_final = make_fulldf(factor_list,exp_matrix)
    exposure_matrix_final.fillna(0,inplace=True)
    xjk_matrix = exposure_matrix_final.as_matrix()
    wigts = matrix(getwigt(wigts_pairs))
    wgted_exp = dot(wigts,xjk_matrix)
    wgted_exp_df = DataFrame(data = wgted_exp.tolist()[0], columns=['exposure'],index=factor_list)
    ##############
    factor_contribution = multiply(wgted_exp,CNE5S_100_FacRet_mat.T)
    ret_cnt = matrix([factor_list,factor_contribution.tolist()[0]]).T

    return_contribution = DataFrame(columns=['Factor','Return'], data = ret_cnt)
    global_lock.acquire()
    print('Date of data:'+target_date_rt+' / Date of Weight:'+target_wgt_dt)
    print('The total weight is: '+str(sum(getwigt(final_wights_pair))))
    if print_middle == True:
        print('\n')
        print('\033[1;46;30m' + target_date_rt + '\033[0m')
        print('\033[1;31m')
        print("####################################################################\nThese are the tables read into function\n####################################################################")
        print('\033[0m')
        print(CNE5S_100_Asset_Exposure.head(2))
        print('*********************************************************')
        print(CHN_LOCALID_Asset_ID.head(2))
        print('*********************************************************')
        print(final_wights.head(2))
        print('*********************************************************')
        print(CNE5S_100_FacRet1.head(2))
    if print_result == True:
        print("######################################\nDAILY RESULT")
        print(return_contribution)
        print('\n\n')
    global_lock.release()
    return (return_contribution,wigts_pairs,wgted_exp_df,dlytotal_specrt,nanlist)

def read_alltables(start_date,end_date):
    daydelta = dt.timedelta(days=1)
    trade_days_init = get_k_data(code = '000300', index = True,\
                    start = date_cvt(end_date), end = date_cvt(start_date)).date
    length = len(trade_days_init)
    crtlen = 0
    target_date_ddm = dt.datetime.strptime(end_date, '%Y%m%d')

    while crtlen < length+1:
        trade_days_list = get_k_data(code = '000300', index = True,\
                    start = date_cvt(target_date_ddm.strftime('%Y%m%d')), end = date_cvt(start_date)).date
        crtlen = len(trade_days_list)
        target_date_ddm = target_date_ddm - daydelta

    trade_dates_list = list(trade_days_list.apply(lambda x : date_cvt(x)))
    # [::-1]
    date_part = dict()
    if length >= 16:
        numcore = 4
        diver = length//numcore

        for ii in range(1,numcore):
            date_part['part'+str(ii)] = trade_dates_list[((ii-1)*diver):((ii)*diver+1)]
        date_part['part'+str(numcore)] = trade_dates_list[((numcore-1)*diver):]

    elif length < 16 and length >= 8:
        numcore = 2
        diver = length//numcore
        
        for ii in range(1,numcore):
            date_part['part'+str(ii)] = trade_dates_list[((ii-1)*diver):((ii)*diver+1)]
        date_part['part'+str(numcore)] = trade_dates_list[((numcore-1)*diver):]

    else:
        date_part['part1'] = trade_dates_list
    print(date_part)
    return(date_part)

def period_accumalate(global_lock, daily_contributions,daily_weights,exposure_matrix_daily,dlytotal_specrt_dict,nandict,\
    read_path_input, trade_dates_list, prod_name, prt_mid,prt_end, index35_test_ind = 'p1f0'):

    futures_pos = read_csv(read_path_input+prod_name+'_futures_info.csv')
    futures_pos['DATE'] = futures_pos.DATE.astype('str')

    asset_allocation = None
    # asset_allocation = read_csv(read_path_input+'asset_allocation.csv')

    index_hs300p = get_k_data(code = '000300', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))
    index_zz500p = get_k_data(code = '000905', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))
    index_sz50p = get_k_data(code = '000016', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))
    for ii in range(1,len(trade_dates_list)):
        dates = trade_dates_list[ii]
        wigt_date = trade_dates_list[ii-1]
        daily_contributions[dates],daily_weights[dates],exposure_matrix_daily[dates],dlytotal_specrt_dict[dates],nandict[dates]\
            = mainfunc(global_lock, read_path_input, dates, wigt_date, futures_pos, asset_allocation, index_hs300p, index_zz500p, index_sz50p, index35_test_ind, prod_name, prt_mid,prt_end)

    return (daily_contributions,daily_weights,exposure_matrix_daily,dlytotal_specrt_dict,nandict)



def mainfunc_period(read_path_input, daily_contributions_table,daily_weights_pairs,dly_period):
    Rt_portfolio = dict()
    for rt_date in list(daily_contributions_table.keys()):
        try:
            daily_return_table = read_table(read_path_input+'CNE5_Daily_Asset_Price.'+rt_date, sep = '|', header=1)
        except:
            print(rt_date)
        else:
            daily_return_table.rename(columns={'!Barrid':'Barrid','DlyReturn%':'DlyReturn'}, inplace = True)
            weights = getwigt(daily_weights_pairs[rt_date])
            wigts_names = getwigt_name(daily_weights_pairs[rt_date])
            daily_rt = []
            for name in wigts_names:
                daily_rt.append(float(daily_return_table[daily_return_table.Barrid == name].DlyReturn))
            Rt_portfolio[rt_date] = dot((array(daily_rt).T), array(weights))
    daily_total_rt = [x*0.01+1 for x in list(Rt_portfolio.values())]
    Rp_period = prod(array(daily_total_rt))
    factor_dicts = dict()
    for rt_date in list(daily_contributions_table.keys()):
        for index, row in daily_contributions_table[rt_date].iterrows():
            try:
                factor_dicts[row[0]]
            except:
                factor_dicts[row[0]] = []
                factor_dicts[row[0]].append(float(row[1]))
            else:
                factor_dicts[row[0]].append(float(row[1]))
    factors_df = DataFrame(factor_dicts) 
    factor_rt_dict = dict()
    for col in factors_df.columns:
        idx = 0
        mupy = 1
        for cell in factors_df[col]:
            Rpt = daily_total_rt[idx]
            mupy = mupy * (Rpt - cell)
            idx += 1
        factor_rt_dict[col] = Rp_period - mupy
    factor_var_dict = dict()
    for cls in factors_df.columns:
        factor_var_dict[cls] = var([x for x in factors_df[cls]])
    
    muspecrt = 1
    idx_spec = 0
    for rt in daily_total_rt:
        muspecrt = muspecrt*(rt - dly_period[idx_spec])
        idx_spec += 1
    specific_rt = Rp_period - muspecrt
    return (factor_rt_dict,Rp_period,specific_rt,factor_var_dict,daily_total_rt,factors_df)


def show_result(read_path_output, date_start, date_end, result, total_return, specific_return_final,
                    factor_var_dict,dly_total_rt,num_trdys,spec_dly,exposure_matrix_daily_table,sufx,index_type,risk_free = 0.035):
    emptydict = dict()
    for i in list(result.keys()):
        emptydict[i] = str(100 * result[i]) + '%'
    df = DataFrame(list(emptydict.values()), index=list(emptydict.keys()))

    exposure_list = (sum(exposure_matrix_daily_table.values()) / len(exposure_matrix_daily_table.keys()))
    final_result = merge(exposure_list, df, left_index=True, right_index=True)
    final_result.columns = ['Exposure', 'Factor Return']

    start_date_m = dt.datetime.strptime(date_start, '%Y%m%d')
    end_date_m = dt.datetime.strptime(date_end, '%Y%m%d')
    date_length = abs((start_date_m - end_date_m).days)
    total_return_year = total_return ** (250 / num_trdys)
    
    if index_type == 'p1f1':
        theta = total_return - 1 - sum(list(result.values())) - specific_return_final
        
    elif (index_type == 'p0f1' or index_type == 'p1f0'):
        theta = total_return - (1+risk_free) ** (date_length / 365) - sum(list(result.values())) - specific_return_final
    else:
        raise PassingError('The passed argument is not as expected in show function')
    total_var = sum(list(factor_var_dict.values()))

    factor_var_dict1 = dict()
    for cls in list(factor_var_dict.keys()):
        factor_var_dict1[cls] = (theta * (factor_var_dict[cls] / total_var)) + result[cls]
    emptydict1 = dict()
    for j in list(result.keys()):
        emptydict1[j] = str(100 * factor_var_dict1[j]) + '%'
    df1 = DataFrame(list(emptydict1.values()), index=list(emptydict1.keys()))

    final_result1 = merge(exposure_list, df1, left_index=True, right_index=True)
    dfvar = DataFrame(list(factor_var_dict.values()), index=list(factor_var_dict.keys()))
    
    final_result2 = merge(final_result1, dfvar, left_index=True, right_index=True)
    
    final_result2.columns = ['Exposure', 'Factor_Return', 'Factor_Var']

    series1 = dly_total_rt.copy()
    series1.insert(0,1)
    net_value = acmultiply(series1)

    otherdict = dict()
    otherdict = {'SpecificR':str(specific_return_final*100)+'%','TotalR':str((total_return-1)*100)+'%','TotalR/Year':str((total_return_year-1)*100)+'%',
                    'TradingDays':str(num_trdys)+' days','Periodlen/day':str(date_length) + ' days','RiskfreeR':str(100*(1+risk_free)**(date_length/365)-100)+'%',
                        'TotalFacR':str(sum(list(factor_var_dict1.values()))*100)+'%','CrosPs':str(theta*100)+'%',
                            'MaxDD': str(max_drawdown(net_value)*100)+' %','Volatility':str((std(dly_total_rt))*100)+' %',
                                 'Vol/Year':str((std(dly_total_rt))*sqrt(250)*100)+' %','SpecRVol/Y':str(std(spec_dly)*sqrt(250)*100)+'%'}
    otherdf = DataFrame(data = {'Result':list(otherdict.values())}, index = list(otherdict.keys()))

    print(otherdf)
    print('\n\nFinal Result with theta concluded is as following')
    print(final_result2)

    output_path_final = read_path_output+index_type+'_'+sufx+'_result_csvs/'
    resultstr = []

    try:
        mkdir(output_path_final)
    except(FileExistsError):
        resultstr.append('The folder used to place results already exists. Please delete the file and try again\n')
    else:
        final_result.to_csv(output_path_final+'Factors_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        final_result2.to_csv(output_path_final+'FactorsTheta_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        otherdf.to_csv(output_path_final+'Summary_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
    
    for keys in otherdict:
        resultstr.append(keys+' '*(20-len(keys))+str(otherdict[keys]))
    
    return resultstr



def mainf(read_start,read_end,sufx,index_mode, risk_free_rate,read_path):
    date_dict = read_alltables(read_start,read_end)
    print("main process runs...")
    start = time.time()
    manager = Manager()
    global_lock = Lock()
    daily_contributions_table_common = manager.dict()
    daily_weights_pairs_common = manager.dict()
    exposure_matrix_daily_table_common = manager.dict()
    dlytotal_specrt_period_common = manager.dict()
    nandict_common = manager.dict()
    p_list = []
    for ix in (date_dict):
        trd_list_part = date_dict[ix]
        p=Process(target=period_accumalate,args=(global_lock,daily_contributions_table_common,daily_weights_pairs_common,\
            exposure_matrix_daily_table_common,dlytotal_specrt_period_common,nandict_common,\
                read_path, trd_list_part, sufx, True, True, index_mode))
        p_list.append(p)
        p.start()
    for res in p_list:
        res.join()

    print("main process runned all lines...")

    daily_contributions_table_common = dict(sorted(daily_contributions_table_common.items(), \
            key=lambda daily_contributions_table_common:dt.datetime.strptime((daily_contributions_table_common[0]), '%Y%m%d')))
    daily_weights_pairs_common = dict(sorted(daily_weights_pairs_common.items(), \
            key=lambda daily_weights_pairs_common:dt.datetime.strptime((daily_weights_pairs_common[0]), '%Y%m%d')))
    exposure_matrix_daily_table_common = dict(sorted(exposure_matrix_daily_table_common.items(), \
            key=lambda exposure_matrix_daily_table_common:dt.datetime.strptime((exposure_matrix_daily_table_common[0]), '%Y%m%d')))
    dlytotal_specrt_period_common = dict(sorted(dlytotal_specrt_period_common.items(), \
            key=lambda dlytotal_specrt_period_common:dt.datetime.strptime((dlytotal_specrt_period_common[0]), '%Y%m%d')))
    nandict_common = dict(sorted(nandict_common.items(), key=lambda nandict_common:dt.datetime.strptime((nandict_common[0]), '%Y%m%d')))

    result,total_return,specific_return_final,factor_var_dict,daily_total_return,factor_rts = mainfunc_period(
        read_path, daily_contributions_table_common,daily_weights_pairs_common,list(dlytotal_specrt_period_common.values()))

    result_summary = show_result(read_path,read_start,read_end, result,total_return,specific_return_final,factor_var_dict,\
        daily_total_return,len(daily_contributions_table_common),list(dlytotal_specrt_period_common.values()),\
            exposure_matrix_daily_table_common,sufx,index_type = index_mode, risk_free = risk_free_rate)
    end = time.time()
    print(end-start)
    return result_summary
if __name__ == "__main__":
    result_summary = mainf('20180427','20180103','hh','p1f1',0.035,'D:/Attibution_Analysis_June/running/')