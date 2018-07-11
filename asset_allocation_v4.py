

#XIAOTIAN 2018/03/29

#CNE5S_100_Asset_Exposure: THE ASSET EXPOSURE FOR EVERY STOCK
#CHN_LOCALID_Asset_ID: THE ASSET ID USED TO MATCH LOCAL ID AND BARRA ID
#CNE5_Daily_Asset_Price
#CNE5S_100_DlyFacRet
#CNE5_100_Asset_DlySpecRet
#future_info.csv


from multiprocessing import Process, Manager, Lock
import datetime as dt
from pandas import DataFrame, concat, read_table, read_csv, merge, isnull, merge
import time
from copy import copy
from tushare import get_k_data
from math import sqrt
from time import strftime
from os import mkdir
from numpy import argmax, maximum, prod, matrix, dot, multiply, array, var, std,  multiply
import matplotlib.pyplot as plt
# multiprocessing.set_start_method('spawn')
class PassingError(Exception): 
        """Error encountered while passing arguments."""
        pass



# Takes a dataframe of index market data and make an additional column of index daily return
def make_idxrt(df):
    closes = list(df.close)
    closes.insert(0,0)
    del closes[-1]
    df['preclose']=closes
    df['idxreturn'] = (df.close/df.preclose)-1
    return df

# Takes two arguments. The first one is a dataframe and the second one is a column name which is going to be checked if there is any null values in that column.
# All observations with nul values in that coluun will be deleted
def Del_Narows(tgt_df, col_name):
    nalist = list(tgt_df.index[isnull(tgt_df[col_name])].values)
    nalist.extend(tgt_df.index[tgt_df[col_name].apply(lambda x: x == 0)].values)
    return tgt_df.drop(nalist)

# It takes raw data of asset weights and make sure the data is well structured for the following uses
def trans_table(tgt_df):
    tgt_df.columns = ['symbol','value','change']
    tgt_df2 = Del_Narows(tgt_df,'value')
    tgt_df2.index = range(len(tgt_df2))
    tgt_df2['value'] = tgt_df2.value.astype('float64')
    tgt_df2['symbol'] = tgt_df2.symbol.astype('str')
    tgt_df2['symbol'] = tgt_df2.symbol.apply(lambda x : ('0'*(6-len(x))+x) if len(x) < 6 else x)

    etfdict = {'IH':0,'IC':0,'IF':0}
    for ii in range(len(tgt_df2)):
        if tgt_df2.loc[ii,'symbol'] == '510050':
            etfdict['IH'] += tgt_df2.loc[ii,'value']

    return tgt_df2[['symbol','value']],etfdict

# It takes a list of data frames which are going to be merged on weights accordingly
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

#
# def sumcols(df1, df2):
#     final_wts = merge(df1, df2, how='outer')
#     final_wts.fillna(0,inplace=True)
#     final_wts['weight'] = final_wts.weight+final_wts.weight
#     return final_wts['Barrid','weight']

# This function takes a local id(CN000001) and return a latest Barra ID
def latest_id(local_id,iddf):
    dfa = iddf[iddf.AssetID.apply(lambda x: x == 'CN' + local_id[:6])]
    return dfa[dfa.EndDate == max(dfa.EndDate)].values[0,0]

#A function used to relate the localids and barraids
def lcl_bra(df1,df2):
    lcldict = DataFrame(columns=['Barrid', 'value'])
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
            lcldict = lcldict.append({'Barrid':latest_id(row[0],df2),'value':row[1]},ignore_index=True)
        count += 1
    total_value = sum(lcldict.value)
    lcldict['weight'] = lcldict.value/total_value
    return lcldict[['Barrid','weight']],total_value,nanlist


# It takes two data frames. The first is raw data of weigts and related local id. The second is CHN_LOCALID_Asset_ID
def lcl_bra_index(df1,df2):
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
    lcldict.weight = lcldict.weight/(sum(df1.weight)-sum(nanweight))
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

# Convert a list of pairs of weights and ids to a dataframe
def DftoPairlist(df):
    emptylist = [0]*len(df)
    for idx in range(len(df)):
        temp = [0,0]
        temp[0] = (df.iloc[idx])[0]
        temp[1] = (df.iloc[idx])[1]
        emptylist[idx] = temp
    return emptylist

# It calculates the max drawdown of this a portfolio
def max_drawdown(timeseries):
    i = argmax(maximum.accumulate(timeseries) - timeseries)
    j = argmax(timeseries[:i])
    return (float(timeseries[i]) / timeseries[j]) - 1.

# It works as the same as numpy.multiply.accumulate
def acmultiply(aseries):
    nalist = []
    for i in range(len(aseries)):
        nalist.append(prod(aseries[:i+1]))
    return nalist

# It converts the dates between two date formats
def date_cvt(astr):
    if len(astr) == 10:
        return astr[0:4]+astr[5:7]+astr[8:10]
    
    elif len(astr) == 8:
        return astr[0:4]+'-'+astr[4:6]+'-'+astr[6:8]
    
    else:
        raise PassingError('The passed argument is not as expected')

# This is the main function used to calculate the daily factor exposure and returns
def mainfunc(global_lock, read_path_input, read_path_spec,target_date_rt, target_wgt_dt, futures_pos, \
            asset_allocation, index_hs300p, index_zz500p,index_sz50p, index35_test, prod_name
             ,print_middle = False,print_result = True):

    CNE5S_100_Asset_Exposure = read_table(read_path_input+'CNE5S_100_Asset_Exposure.'+target_date_rt, sep = '|', header=2)

    CHN_LOCALID_Asset_ID = read_csv(read_path_input+'CHN_LOCALID_Asset_ID.'+target_date_rt, sep = '|', header=1)
    CNE5S_100_DlyFacRet = read_table(read_path_input+'CNE5S_100_DlyFacRet.'+target_date_rt, sep = '|', header=2)
    
    CNE5_100_Asset_DlySpecRet = read_table(read_path_input+'CNE5_100_Asset_DlySpecRet.'+target_date_rt, sep = '|', header=2)
    CNE5_100_Asset_DlySpecRet.rename(columns={'!Barrid':'Barrid'}, inplace = True)
    hs300_read = read_csv(read_path_input+'hs300w_'+target_wgt_dt+'.csv')
    zz500_read = read_csv(read_path_input+'zz500w_'+target_wgt_dt+'.csv')
    sz50_read = read_csv(read_path_input+'sz50w_'+target_wgt_dt+'.csv')

    hs300,nanlisths = lcl_bra_index(hs300_read,CHN_LOCALID_Asset_ID)
    zz500,nanlistzz = lcl_bra_index(zz500_read,CHN_LOCALID_Asset_ID)
    sz50,nanlistsz = lcl_bra_index(sz50_read,CHN_LOCALID_Asset_ID)

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

    total_portfilo_allocation = read_csv(read_path_spec+prod_name+'_netvalues.csv')

    total_portfilo_value = total_portfilo_allocation[total_portfilo_allocation.DATE.apply(lambda x: str(x) == date_cvt(target_wgt_dt))].NV.values[0]

    wgt_CH,etfdict = trans_table(read_csv(read_path_spec+prod_name+'_shares_'+target_wgt_dt+'.csv'))

    daily_wigt_dist = dict()

    if index35_test == 'p1f0' or ic_pos == if_pos == ih_pos == 0:
        wgt_CH1,total_share_value,nanlist = lcl_bra(wgt_CH,CHN_LOCALID_Asset_ID)

        if sum(etfdict.values()) != 0:

            futures_table = {'IC':zz500,'IF':hs300,'IH':sz50}

            wgt_CH1['weight'] = wgt_CH1.weight * (total_share_value / total_portfilo_value)
            final_stocks_weight = wgt_CH1

            futures_table_edited = dict()
            futures_subwigt = dict()

            for fts in etfdict:
                result_df = futures_table[fts]
                reuslt_wigt = etfdict[fts]
                
                if etfdict[fts]>0:
                    result_df['weight'] = abs((reuslt_wigt/total_portfilo_value)) * result_df.weight

                elif etfdict[fts]<0:
                    result_df['weight'] = -abs((reuslt_wigt/total_portfilo_value)) * result_df.weight

                elif etfdict[fts]==0:
                    result_df['weight'] = 0 * result_df.weight
                    
                else:
                    raise Exception
                futures_table_edited[fts] = result_df
                futures_subwigt[fts] = sum(result_df.weight)

            hs300 = futures_table_edited['IF']
            zz500 = futures_table_edited['IC']
            sz50 = futures_table_edited['IH']

            input_wigt_list = [wgt_CH1, hs300, zz500, sz50]

            idxweights_table = idx_weight(input_wigt_list)

            final_wights = idxweights_table

            daily_wigt_dist = {'Portfolio':abs(total_share_value / total_portfilo_value), 'IH':abs(sum(etfdict.values())/total_portfilo_value),'IC':0,'IF':0}

        else:
            final_wights = wgt_CH1
            final_stocks_weight = final_wights
            futures_subwigt = []
            daily_wigt_dist = {'Portfolio':1,'IH':0,'IC':0,'IF':0}

    elif index35_test == 'p1f1' and not(ic_pos == if_pos == ih_pos == 0):

        wgt_CH1,total_share_value,nanlist = lcl_bra(wgt_CH,CHN_LOCALID_Asset_ID)

        ic_weight = ic_pos*ic_price*200
        if_weight = if_pos*if_price*300
        ih_weight = ih_pos*ih_price*300

        wgt_CH1['weight'] =  wgt_CH1.weight*(total_share_value/total_portfilo_value)

        futures_dict = {'IC':ic_weight,'IF':if_weight,'IH':ih_weight}

        for fut in etfdict:
            futures_dict[fut] += etfdict[fut]
        daily_wigt_dist = dict()
        futures_table = {'IC':zz500,'IF':hs300,'IH':sz50}
        futures_table_edited = dict()
        futures_subwigt = dict()
        for fts in futures_dict:
            result_df = futures_table[fts]
            reuslt_wigt = futures_dict[fts]
                
            if reuslt_wigt>0:
                result_df['weight'] = abs((reuslt_wigt/total_portfilo_value)) * result_df.weight
                daily_wigt_dist[fts] = abs((reuslt_wigt/total_portfilo_value))
            elif reuslt_wigt<0:
                result_df['weight'] = -abs((reuslt_wigt/total_portfilo_value)) * result_df.weight
                daily_wigt_dist[fts] = -abs((reuslt_wigt/total_portfilo_value))
            elif reuslt_wigt==0:
                result_df['weight'] = 0 * result_df.weight
                daily_wigt_dist[fts] = 0
            else:
                raise Exception
            futures_table_edited[fts] = result_df
            futures_subwigt[fts] = sum(result_df.weight)

        hs300 = futures_table_edited['IF']
        zz500 = futures_table_edited['IC']
        sz50 = futures_table_edited['IH']

        final_stocks_weight = wgt_CH1.copy()

        input_wigt_list = [wgt_CH1, hs300, zz500, sz50]

        idxweights_table = idx_weight(input_wigt_list)

        final_wights = idxweights_table

        nanlist.extend(nanlisths)
        nanlist.extend(nanlistzz)
        nanlist.extend(nanlistsz)

        daily_wigt_dist['Portfolio'] = (total_share_value/total_portfilo_value)

    elif index35_test == 'p0f1':
        ic_weight = ic_pos*ic_price*200
        if_weight = if_pos*if_price*300
        ih_weight = ih_pos*ih_price*300

        hs300['weight'] = abs((if_weight/total_portfilo_value)) * hs300.weight
        zz500['weight'] = abs((ic_weight/total_portfilo_value)) * zz500.weight
        sz50['weight'] = abs((ih_weight/total_portfilo_value)) * sz50.weight

        input_wigt_list = [hs300, zz500, sz50]

        idxweights_table = idx_weight(input_wigt_list)
        final_wights = idxweights_table
        
        nanlist =[]
        nanlist.extend(nanlisths)
        nanlist.extend(nanlistzz)
        nanlist.extend(nanlistsz)
        wgt_CHx = wgt_CH
        wgt_CHx['weight'] = wgt_CHx.weight*0
        final_stocks_weight = wgt_CHx
        daily_wigt_dist = {'Portfolio':0,'IC':(ic_weight/total_portfilo_value),'IF':(if_weight/total_portfilo_value),'IH':(ih_weight/total_portfilo_value)}
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
        try:
            spec_rt = float(CNE5_100_Asset_DlySpecRet[CNE5_100_Asset_DlySpecRet.Barrid == pair[0]].SpecificReturn.values[0])
        except:
            print(pair[0]+' '+target_date_rt+' empty specific return')
        else:
            dlytotal_specrt += 0.01 * spec_rt * pair[1]
    #Amendation to the factor returns column
    #Get the factor list name
        
    CNE5S_100_FacRet_mat = matrix(list(CNE5S_100_FacRet1.DlyReturn)).T
    factor_list = list(CNE5S_100_FacRet1.Factor)
    exp_matrix = make_expmat(wigts_pairs,CNE5S_100_Asset_Exposure,factor_list)
    exposure_matrix_final = make_fulldf(factor_list,exp_matrix)
    exposure_matrix_final.fillna(0,inplace=True)
    xjk_matrix = exposure_matrix_final.values
    wigts = matrix(getwigt(wigts_pairs))
    wgted_exp = dot(wigts,xjk_matrix)
    wgted_exp_df = DataFrame(data = wgted_exp.tolist()[0], columns=['exposure'],index=factor_list)
    ##############
    factor_contribution = multiply(wgted_exp,CNE5S_100_FacRet_mat.T)
    ret_cnt = matrix([factor_list,factor_contribution.tolist()[0]]).T

    return_contribution = DataFrame(columns=['Factor','Return'], data = ret_cnt)
    return_contribution['Return'] = return_contribution.Return.astype('float64')
    global_lock.acquire()
    print('Date of data:'+target_date_rt+' / Date of Weight:'+target_wgt_dt)
    print('The total weight is: '+str(sum(getwigt(final_wights_pair))))
    print('The weight distribution:')
    print(daily_wigt_dist)
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

    return (return_contribution,final_stocks_weight,wgted_exp_df,dlytotal_specrt,nanlist,daily_wigt_dist)

# This function takes the start date and end date and returns a few list of dates used in mainfunc
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
    return(date_part,trade_dates_list)


def period_accumalate(global_lock, daily_contributions,daily_weights,exposure_matrix_daily,dlytotal_specrt_dict,nandict,daily_wigt_dist,\
    read_path_input,read_path_spec, trade_dates_list, prod_name, prt_mid,prt_end, index35_test_ind = 'p1f0'):

    futures_pos = read_csv(read_path_spec+prod_name+'_futures_info.csv')
    futures_pos['DATE'] = futures_pos.DATE.astype('str')

    asset_allocation = None
    # asset_allocation = read_csv(read_path_input+'asset_allocation.csv')

    index_hs300p = get_k_data(code = '000300', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))
    index_zz500p = get_k_data(code = '000905', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))
    index_sz50p = get_k_data(code = '000016', index = True, start = date_cvt(trade_dates_list[0]), end = date_cvt(trade_dates_list[-1]))

    index_ic = (index_zz500p)
    index_if = (index_hs300p)
    index_ih = (index_sz50p)

    for ii in range(1,len(trade_dates_list)):
        dates = trade_dates_list[ii]
        wigt_date = trade_dates_list[ii-1]
        daily_contributions[dates],daily_weights[dates],exposure_matrix_daily[dates],dlytotal_specrt_dict[dates],nandict[dates],daily_wigt_dist[dates]\
            = mainfunc(global_lock, read_path_input, read_path_spec, dates, wigt_date, futures_pos, \
                       asset_allocation, index_if, index_ic, index_ih, index35_test_ind, prod_name, prt_mid,prt_end)

    return (daily_contributions,daily_weights,exposure_matrix_daily,dlytotal_specrt_dict,nandict,daily_wigt_dist)


# It takes some daily results calculated above and calculate the analysis result for the whole period.
def mainfunc_period(read_path_input, daily_contributions_table,daily_weights_pairs,dly_period,daily_wigt_dist_total,trade_dates_list,read_path_spec,prod_name):
    Rt_portfolio = dict()
    print(trade_dates_list)

    print(daily_wigt_dist_total)

    index_hs300p = get_k_data(code='000300', index=True, start=date_cvt(trade_dates_list[0]),
                                 end=date_cvt(trade_dates_list[-1]))
    index_zz500p = get_k_data(code='000905', index=True, start=date_cvt(trade_dates_list[0]),
                                 end=date_cvt(trade_dates_list[-1]))
    index_sz50p = get_k_data(code='000016', index=True, start=date_cvt(trade_dates_list[0]),
                                end=date_cvt(trade_dates_list[-1]))

    index_ic = make_idxrt(index_zz500p)
    index_if = make_idxrt(index_hs300p)
    index_ih = make_idxrt(index_sz50p)

    stocks_dict = dict()
    futures_dict = dict()

    total_portfilo_allocation = read_csv(read_path_spec+prod_name+'_netvalues.csv')


    daily_real_returns = dict()

#################################################
    for datei in range(1,len(trade_dates_list)):

        rt_date = trade_dates_list[datei]
        wigt_date = trade_dates_list[datei-1]

        total_portfilo_value = total_portfilo_allocation[total_portfilo_allocation.DATE.apply(lambda x: str(x) == date_cvt(wigt_date))].NV.values[0]

        total_portfilo_value_today = total_portfilo_allocation[total_portfilo_allocation.DATE.apply(lambda x: str(x) == date_cvt(rt_date))].NV.values[0]

        total_portfilo_real_return = total_portfilo_value_today/total_portfilo_value

        daily_real_returns[rt_date] = total_portfilo_real_return

        try:
            daily_return_table = read_table(read_path_input+'CNE5_Daily_Asset_Price.'+rt_date, sep = '|', header=1)
        except:
            print(rt_date)
        else:
            daily_return_table.rename(columns={'!Barrid':'Barrid','DlyReturn%':'DlyReturn'}, inplace = True)
            weights = (daily_weights_pairs[rt_date])
            return_df = merge(weights,daily_return_table, on='Barrid', how = 'left')
            if return_df.isnull().values.any():
                return_df = return_df.dropna()
                print('missing return on ' + rt_date)
                #####################
            ifrt = index_if[index_if.date == date_cvt(rt_date)].idxreturn.values[0]
            icrt = index_ic[index_ic.date == date_cvt(rt_date)].idxreturn.values[0]
            ihrt = index_ih[index_ih.date == date_cvt(rt_date)].idxreturn.values[0]

            daily_wigt_dist = daily_wigt_dist_total[rt_date]

            stock_weight = daily_wigt_dist['Portfolio']
            icpos = daily_wigt_dist['IC']
            ifpos = daily_wigt_dist['IF']
            ihpos = daily_wigt_dist['IH']

            future_rts = {'IC':icpos*icrt,'IF':ifpos*ifrt,'IH':ihpos*ihrt}

            positive_futures = 0
            negetive_futures = 0

            for fts in future_rts:
                if daily_wigt_dist[fts]>0:
                    positive_futures += future_rts[fts]
                if daily_wigt_dist[fts]<0:
                    negetive_futures += future_rts[fts]

            Rt_portfolio[rt_date] = sum(
                return_df.DlyReturn * return_df.weight) * stock_weight * 0.01 + positive_futures + negetive_futures
            stocks_dict[rt_date] = sum(return_df.DlyReturn*return_df.weight)*stock_weight*0.01 + positive_futures
            futures_dict[rt_date] = negetive_futures
            #########################
    print(Rt_portfolio)

    daily_total_rt = [x+1 for x in list(Rt_portfolio.values())]

    print(daily_real_returns)

    stocks_return = prod(array([x+1 for x in list(stocks_dict.values())]))
    futures_return = prod(array([x+1 for x in list(futures_dict.values())]))

    print('stocks_return')
    print(stocks_dict)
    print(stocks_return)

    print('futures_return')
    print(futures_dict)
    print(futures_return)

    print('Method W')

    Rp_period = 1 + stocks_return-1 + futures_return-1

    # Rp_period = prod(daily_total_rt)

    print(Rp_period)
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


# It takes all the result calculated and show the final result in certain formats
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
    print('raw_factors')
    print(sum(list(result.values())))
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

    output_path_final = read_path_output+index_type+'_'+date_start+'_'+date_end+'_'+sufx+'_result_csvs/'
    resultstr = []

    try:
        mkdir(output_path_final)
    except(FileExistsError):
        resultstr.append('The folder used to place results already exists. Please delete the file and try again\n')
        hrmtsds = ((time.localtime(time.time()))[3:6])
        timestr = str(hrmtsds[0])+'h'+str(hrmtsds[1])+'m'
        output_path_final = read_path_output+index_type+'_'+date_start+'_'+date_end+'_'+sufx+'_'+timestr+'_result_csvs/'

        mkdir(output_path_final)

        final_result.to_csv(output_path_final+'Factors_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        final_result2.to_csv(output_path_final+'FactorsTheta_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        otherdf.to_csv(output_path_final+'Summary_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
    else:
        final_result.to_csv(output_path_final+'Factors_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        final_result2.to_csv(output_path_final+'FactorsTheta_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
        otherdf.to_csv(output_path_final+'Summary_'+sufx+'_'+str(strftime("%Y%m%d"))+'.csv')
    
    for keys in otherdict:
        resultstr.append(keys+' '*(20-len(keys))+str(otherdict[keys]))
    
    return resultstr



# This is a function managing different processes
def mainf(read_start,read_end,sufx,index_mode, risk_free_rate,read_path_data,read_path_spec):
    (date_dict,trade_dates_list) = read_alltables(read_start,read_end)
    print("main process runs...")
    start = time.time()
    manager = Manager()
    global_lock = Lock()
    daily_contributions_table_common = manager.dict()
    daily_weights_pairs_common = manager.dict()
    exposure_matrix_daily_table_common = manager.dict()
    dlytotal_specrt_period_common = manager.dict()
    nandict_common = manager.dict()
    daily_wigt_dist_common = manager.dict()
    p_list = []
    for ix in (date_dict):
        trd_list_part = date_dict[ix]
        p=Process(target=period_accumalate,args=(global_lock,daily_contributions_table_common,daily_weights_pairs_common,\
            exposure_matrix_daily_table_common,dlytotal_specrt_period_common,nandict_common,daily_wigt_dist_common,\
                read_path_data,read_path_spec, trd_list_part, sufx, True, True, index_mode))
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
    daily_wigt_dist_common = dict(sorted(daily_wigt_dist_common.items(), \
            key=lambda daily_wigt_dist_common: dt.datetime.strptime((daily_wigt_dist_common[0]), '%Y%m%d')))

    result,total_return,specific_return_final,factor_var_dict,daily_total_return,factor_rts = mainfunc_period(
        read_path_data, daily_contributions_table_common,daily_weights_pairs_common,list(dlytotal_specrt_period_common.values()),daily_wigt_dist_common,trade_dates_list,read_path_spec,sufx)

    result_summary = show_result(read_path_spec,read_start,read_end, result,total_return,specific_return_final,factor_var_dict,\
        daily_total_return,len(daily_contributions_table_common),list(dlytotal_specrt_period_common.values()),\
            exposure_matrix_daily_table_common,sufx,index_type = index_mode, risk_free = risk_free_rate)
    end = time.time()
    print(end-start)

    factor_types_return = [(sum(x['Return'][:10]),sum(x['Return'][10:]), y) for x, y in zip(daily_contributions_table_common.values(), dlytotal_specrt_period_common.values())]
    style_factors =  multiply.accumulate([x[0]+1 for x in factor_types_return])
    industry_factors =  multiply.accumulate([x[1]+1 for x in factor_types_return])
    assetslct_factors =  multiply.accumulate([x[2]+1 for x in factor_types_return])
    line1, = plt.plot(style_factors, label="Style Factors")
    line2, = plt.plot(industry_factors, label="Industry Factors")
    line3, = plt.plot(assetslct_factors, label="Asset Selection Factors")
    first_legend = plt.legend(handles=[line1,line2,line3], loc=1)
    plt.show()

    return result_summary

if __name__ == "__main__":
    result_summary = mainf('20180629','20180201','jxhy','p1f1',0.035,'D:/datasets/','D:/jupyter_work/jxhy_info/')
    # aax = [20180103,20180131,20180201,20180228,20180301,20180330,20180402,20180427,20180502,20180531,20180601,20180621]
    # for ii in range(6):
    #     ass = (aax[ii*2],aax[ii*2+1])
    #     result_summary = mainf(str(ass[1]),str(ass[0]),'hh','p1f1',0.035,'D:/Attibution_Analysis_June/running/')


    