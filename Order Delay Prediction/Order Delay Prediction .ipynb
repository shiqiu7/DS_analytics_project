{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Step1: define the problem\n",
    "1. The business objective: identify factors and 2. build a model to predict if order will be fulfilled on time or delay.\n",
    "2. The problem is supurvised learning, and a classfication problem\n",
    "3. performance evaluation: ROC/AUC,recall/precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#import packages\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataframe dimensions: (12065, 58)\n"
     ]
    }
   ],
   "source": [
    "#import data\n",
    "df=pd.read_csv('Consolidated data.csv')\n",
    "print('Dataframe dimensions:', df.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Step2: Data Cleaning and manipulation before data exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#subset only delivered and know labels' data\n",
    "df_d=df[(df['SHP_PACK_DLVRY_STS_KEY']==6)&(df['SPDPD_KEY']!=1)&(df['SPDPD_KEY']!=10)]\n",
    "df_d=df_d.dropna(subset=['SPDPD_KEY'])\n",
    "df_d['ontime_ornot']=df_d['SPDPD_KEY']\n",
    "df_d['ontime_ornot']=df_d['ontime_ornot'].replace([2,4],1).replace([3,5,6,7,8,9],0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.07815777473780205"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#ratio of delay:\n",
    "1-sum(df_d['ontime_ornot'])/len(df_d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop clearly not relevant columns\n",
    "df_d=df_d.drop(['CREATED_TS','CREATED_USER','UPDATED_TS','UPDATED_USER','FRAUD_CHECK_REQD_IND','CORP_ORDER_IND','CONSUMER_CNTCT_ID','TENANT_ASSOC_ID'],axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#unify null\n",
    "df_d=df_d.replace('?',np.nan)\n",
    "df_d=df_d.replace('NULL',np.nan)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clean time stamp\n",
    "for i in (5,6,7,9,10,12,15,16,17):\n",
    "    for j in range(len(df_d)):\n",
    "        if type(df_d.iloc[j,i]) is str:\n",
    "            df_d.iloc[j,i]=df_d.iloc[j,i][:-13]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clean SHIPPED_PACKAGE_PRICE_AMT\n",
    "df_d['SHIPPED_PACKAGE_PRICE_AMT']=df_d['SHIPPED_PACKAGE_PRICE_AMT'].str.replace(\",\",\"\").astype(float)\n",
    "df_d['SHIPPED_PACKAGE_PRICE_AMT']=df_d['SHIPPED_PACKAGE_PRICE_AMT'].astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#change time stamp date type\n",
    "\n",
    "for i in ('ORDER_TS','EXPECTED_DELIVERY_TS','LEAST_DELIVERY_TS','DETERMINED_DELIVERY_TS','EXPECTED_SHIPMENT_TS','CARRIER_EXPECTED_DELIVERY_TS','CARGO_DEPARTED_TS','INITIAL_DLVR_ATMPT_TS','FINAL_DELIVERY_ATMPT_TS','ACTUAL_DELIVERY_TS','ORDER_REL_TS','ORDER_PLACED_TS'):\n",
    "    df_d[i]=pd.to_datetime(df_d[i])\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#change index\n",
    "df_d.index=range(len(df_d))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(8772, 51)\n",
      "(2193, 51)\n"
     ]
    }
   ],
   "source": [
    "#create test data: stratified sampling\n",
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) \n",
    "for train_index, test_index in sp.split(df_d, df_d['ontime_ornot']):\n",
    "        df_train = df_d.loc[train_index]\n",
    "        df_test = df_d.loc[test_index]\n",
    "print(df_train.shape)\n",
    "print(df_test.shape)\n",
    "combine=[df_train,df_test]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Step3: Data exploration\n",
    "(more on the report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DSTRBTR_KEY\n",
      "1    0.094772\n",
      "2    0.034188\n",
      "3    0.071545\n",
      "4    0.069767\n",
      "Name: ontime_ornot, dtype: float64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1a1471add8>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# explore analysis\n",
    "#1. distributor feature:\n",
    "a=1-df_train['ontime_ornot'].groupby(df_train['DSTRBTR_KEY']).apply(np.mean)\n",
    "print(a)\n",
    "sns.barplot(a.index,a.values)\n",
    "#conclusion: distributor is a factor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "order_month\n",
       "1.0     0.166667\n",
       "1.5     0.036304\n",
       "2.0     0.073801\n",
       "2.5     0.106897\n",
       "3.0     0.114667\n",
       "3.5     0.090909\n",
       "4.0     0.086957\n",
       "4.5     0.037415\n",
       "5.0     0.038462\n",
       "5.5     0.071611\n",
       "6.0     0.048980\n",
       "6.5     0.072727\n",
       "7.0     0.052632\n",
       "7.5     0.075975\n",
       "8.0     0.084592\n",
       "8.5     0.043590\n",
       "9.0     0.123393\n",
       "9.5     0.057751\n",
       "10.0    0.037572\n",
       "10.5    0.106017\n",
       "11.0    0.066265\n",
       "11.5    0.113483\n",
       "12.0    0.051667\n",
       "12.5    0.000000\n",
       "Name: ontime_ornot, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#2. feature engineering: order month (half month)\n",
    "combine=[df_train,df_test]\n",
    "for data in combine:\n",
    "    data['order_month']=''\n",
    "    for j in range(len(data)):\n",
    "        if data.iloc[j,5].day in range(15):\n",
    "            a=0\n",
    "        else:\n",
    "            a=0.5\n",
    "        data.iloc[j,51]=data.iloc[j,5].month+a\n",
    "a=1-df_train['ontime_ornot'].groupby(df_train['order_month']).apply(np.mean)\n",
    "sns.barplot(a.index,a.values)\n",
    "a\n",
    "#conclusion: seasonality is an important feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SHIPPED_PACKAGE_PRICE_AMT</th>\n",
       "      <th>TOT_AMT</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ontime_ornot</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>36.350598</td>\n",
       "      <td>82.881412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>30.382483</td>\n",
       "      <td>88.327062</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              SHIPPED_PACKAGE_PRICE_AMT    TOT_AMT\n",
       "ontime_ornot                                      \n",
       "0                             36.350598  82.881412\n",
       "1                             30.382483  88.327062"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#3. SHIPPED_PACKAGE_PRICE_AMT : more on tableau\n",
    "df_train[['SHIPPED_PACKAGE_PRICE_AMT','TOT_AMT']].groupby(df_train['ontime_ornot']).apply(np.mean)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Expected delivery time for on time\n",
      "count                      8086\n",
      "mean     5 days 17:05:53.476131\n",
      "std      3 days 16:21:17.901670\n",
      "min             0 days 21:48:06\n",
      "25%      2 days 18:49:35.750000\n",
      "50%             4 days 15:18:21\n",
      "75%      8 days 01:53:15.250000\n",
      "max            39 days 21:26:48\n",
      "Name: expected delivery time, dtype: object\n",
      "------------\n",
      "Expected delivery time for delays\n",
      "count                       686\n",
      "mean     4 days 08:31:47.288629\n",
      "std      2 days 18:43:33.584231\n",
      "min             0 days 22:47:52\n",
      "25%      2 days 07:12:52.250000\n",
      "50%             3 days 05:24:30\n",
      "75%      5 days 05:12:17.750000\n",
      "max            14 days 19:29:35\n",
      "Name: expected delivery time, dtype: object\n"
     ]
    }
   ],
   "source": [
    "#4. feature engineering use timestamp-----expected delivery time\n",
    "df_train['expected delivery time']=df_train['EXPECTED_DELIVERY_TS']-df_train['ORDER_TS']\n",
    "df_test['expected delivery time']=df_test['EXPECTED_DELIVERY_TS']-df_test['ORDER_TS']\n",
    "print(\"Expected delivery time for on time\")\n",
    "print(df_train[df_train['ontime_ornot']==1]['expected delivery time'].describe())\n",
    "print('------------')\n",
    "print(\"Expected delivery time for delays\")\n",
    "print(df_train[df_train['ontime_ornot']==0]['expected delivery time'].describe())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#5. states----on tableau\n",
    "#rename state column\n",
    "\n",
    "df_train=df_train.rename(columns={'CITY_STATE_INFO_Vcol':'state'})  \n",
    "df_test=df_test.rename(columns={'CITY_STATE_INFO_Vcol':'state'}) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Expected delivery range for on time\n",
      "count                      8085\n",
      "mean     1 days 00:28:19.146567\n",
      "std      2 days 01:52:10.152815\n",
      "min             0 days 00:00:00\n",
      "25%             0 days 00:00:00\n",
      "50%             0 days 00:00:00\n",
      "75%             0 days 00:00:00\n",
      "max            15 days 00:00:00\n",
      "Name: delivery time range, dtype: object\n",
      "------------\n",
      "Expected delivery range for delay\n",
      "count                       686\n",
      "mean     0 days 09:16:16.093294\n",
      "std      1 days 08:17:55.925594\n",
      "min             0 days 00:00:00\n",
      "25%             0 days 00:00:00\n",
      "50%             0 days 00:00:00\n",
      "75%             0 days 00:00:00\n",
      "max             6 days 00:00:00\n",
      "Name: delivery time range, dtype: object\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>delivery time range</th>\n",
       "      <th>expected delivery time</th>\n",
       "      <th>ontime_ornot</th>\n",
       "      <th>SHIPPED_PACKAGE_PRICE_AMT</th>\n",
       "      <th>TOT_AMT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ontime_ornot</th>\n",
       "      <td>0.082906</td>\n",
       "      <td>0.099334</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.026392</td>\n",
       "      <td>0.012059</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>expected delivery time</th>\n",
       "      <td>0.726382</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.099334</td>\n",
       "      <td>0.084179</td>\n",
       "      <td>-0.001928</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>delivery time range</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.726382</td>\n",
       "      <td>0.082906</td>\n",
       "      <td>-0.015827</td>\n",
       "      <td>-0.001453</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TOT_AMT</th>\n",
       "      <td>-0.001453</td>\n",
       "      <td>-0.001928</td>\n",
       "      <td>0.012059</td>\n",
       "      <td>-0.023263</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SHIPPED_PACKAGE_PRICE_AMT</th>\n",
       "      <td>-0.015827</td>\n",
       "      <td>0.084179</td>\n",
       "      <td>-0.026392</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.023263</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           delivery time range  expected delivery time  \\\n",
       "ontime_ornot                          0.082906                0.099334   \n",
       "expected delivery time                0.726382                1.000000   \n",
       "delivery time range                   1.000000                0.726382   \n",
       "TOT_AMT                              -0.001453               -0.001928   \n",
       "SHIPPED_PACKAGE_PRICE_AMT            -0.015827                0.084179   \n",
       "\n",
       "                           ontime_ornot  SHIPPED_PACKAGE_PRICE_AMT   TOT_AMT  \n",
       "ontime_ornot                   1.000000                  -0.026392  0.012059  \n",
       "expected delivery time         0.099334                   0.084179 -0.001928  \n",
       "delivery time range            0.082906                  -0.015827 -0.001453  \n",
       "TOT_AMT                        0.012059                  -0.023263  1.000000  \n",
       "SHIPPED_PACKAGE_PRICE_AMT     -0.026392                   1.000000 -0.023263  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#determined-least\n",
    "df_train['delivery time range']=df_train['DETERMINED_DELIVERY_TS']-df_train['LEAST_DELIVERY_TS']\n",
    "df_test['delivery time range']=df_test['DETERMINED_DELIVERY_TS']-df_train['LEAST_DELIVERY_TS']\n",
    "print(\"Expected delivery range for on time\")\n",
    "print(df_train[df_train['ontime_ornot']==1]['delivery time range'].describe())\n",
    "print('------------')\n",
    "print(\"Expected delivery range for delay\")\n",
    "print(df_train[df_train['ontime_ornot']==0]['delivery time range'].describe())\n",
    "#analysis: this is an effective factor, but need to check correlation\n",
    "#correlation between quantitative attributes\n",
    "df_train[(df_train['delivery time range'].notnull())&(df_train['TOT_AMT'].notnull())][['delivery time range','expected delivery time','ontime_ornot','SHIPPED_PACKAGE_PRICE_AMT','TOT_AMT']].astype(int).corr().sort_values(by='ontime_ornot',ascending=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Step4: data manipulation after data exploration\n",
    "    1. one-hot encoding for categorical features\n",
    "    2. feature scaling for quantitative features\n",
    "    3. complete null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#extract factors\n",
    "df_train_model=df_train[['DSTRBTR_KEY','state','order_month','SHIPPED_PACKAGE_PRICE_AMT','expected delivery time','delivery time range','TOT_AMT','ontime_ornot']]\n",
    "df_test_model=df_test[['DSTRBTR_KEY','state','order_month','SHIPPED_PACKAGE_PRICE_AMT','expected delivery time','delivery time range','TOT_AMT','ontime_ornot']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 8772 entries, 100 to 7050\n",
      "Data columns (total 8 columns):\n",
      "DSTRBTR_KEY                  8772 non-null int64\n",
      "state                        8772 non-null object\n",
      "order_month                  8772 non-null object\n",
      "SHIPPED_PACKAGE_PRICE_AMT    8772 non-null float64\n",
      "expected delivery time       8772 non-null timedelta64[ns]\n",
      "delivery time range          8771 non-null timedelta64[ns]\n",
      "TOT_AMT                      8712 non-null float64\n",
      "ontime_ornot                 8772 non-null int64\n",
      "dtypes: float64(2), int64(2), object(2), timedelta64[ns](2)\n",
      "memory usage: 936.8+ KB\n"
     ]
    }
   ],
   "source": [
    "df_train_model.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#one-hot encoding for distributor, state and month\n",
    "df_train_model=pd.get_dummies(df_train_model, columns=['DSTRBTR_KEY','order_month','state'])\n",
    "df_test_model=pd.get_dummies(df_test_model, columns=['DSTRBTR_KEY','order_month','state'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_model['TOT_AMT']=df_train_model['TOT_AMT'].fillna(df_train_model['TOT_AMT'].median())\n",
    "df_test_model['TOT_AMT']=df_test_model['TOT_AMT'].fillna(df_test_model['TOT_AMT'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py:617: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/anaconda3/lib/python3.6/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, **fit_params).transform(X)\n",
      "/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py:617: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/anaconda3/lib/python3.6/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, **fit_params).transform(X)\n"
     ]
    }
   ],
   "source": [
    "#scalling for SHIPPED_PACKAGE_PRICE_AMT,OT_AMT, and expected delivery time\n",
    "df_train_model['expected delivery time']=df_train_model['expected delivery time'].astype(int)\n",
    "df_test_model['expected delivery time']=df_test_model['expected delivery time'].astype(int)\n",
    "df_train_model['delivery time range']=df_train_model['delivery time range'].astype(int)\n",
    "df_test_model['delivery time range']=df_test_model['delivery time range'].astype(int)\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "df_train_model.iloc[:,0:4]= StandardScaler().fit_transform(df_train_model.iloc[:,0:4])\n",
    "df_test_model.iloc[:,0:4]= StandardScaler().fit_transform(df_test_model.iloc[:,0:4])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "state\n",
      "AE    9\n",
      "AK    3\n",
      "HI    4\n",
      "PR    1\n",
      "VI    9\n",
      "WY    6\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#remove states with too few observations\n",
    "a=pd.Series(df_train.index).groupby(df_train['state']).agg('count')\n",
    "print(a[a<10])\n",
    "df_train_model=df_train_model.drop(['state_AE','state_AK','state_HI','state_PR','state_VI','state_WY'],axis=1)\n",
    "df_test_model=df_test_model.drop(['state_AK','state_HI','state_VI','state_WY'],axis=1)\n",
    "#add IA so that train and test data can be same\n",
    "df_test_model['state_IA']=0\n",
    "df_test_model['state_DC']=0\n",
    "df_test_model=df_test_model[list(df_train_model.columns)]\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_model=df_train_model.drop('Unnamed: 0',axis=1)\n",
    "df_test_model=df_test_model.drop('Unnamed: 0',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_model.to_csv('df_train_model.csv')\n",
    "df_test_model.to_csv('df_test_model.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_model=pd.read_csv('df_train_model.csv')\n",
    "df_test_model=pd.read_csv('df_test_model.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Step 5:  find the best model\n",
    "1. Models used: Random Forest, Xgboost,KNN, Logistic regression\n",
    "2. Evalution metrics: I use AUC/ROC because it is animblance classification problem\n",
    "3. Avoid overfitting techniques: K-fold cross validation, tuning hyperparameters\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "#split into X and y\n",
    "X=df_train_model.drop('ontime_ornot',axis=1)\n",
    "y=df_train_model['ontime_ornot']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best: 0.816257 using {'max_depth': 19, 'max_features': 'log2', 'n_estimators': 450}\n"
     ]
    }
   ],
   "source": [
    "#tune randanm forest\n",
    "#hyperparameter: number of trees, max depth, and max features\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "model = ensemble.RandomForestClassifier()\n",
    "n_estimators = range(50, 500, 50)\n",
    "max_depth=range(3,20,1)\n",
    "max_features=('sqrt','log2')\n",
    "param_grid = dict(n_estimators=n_estimators,max_features =max_features,max_depth=max_depth)\n",
    "kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)\n",
    "grid_search = GridSearchCV(model, param_grid, scoring=\"roc_auc\", n_jobs=-1, cv=kfold)\n",
    "grid_result = grid_search.fit(X, y)\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best: 0.806741 using {'max_depth': 9, 'n_estimators': 250}\n"
     ]
    }
   ],
   "source": [
    "#tune xgboost\n",
    "#tune number of trees, max depth\n",
    "import xgboost as xgb\n",
    "model = xgb.XGBClassifier()\n",
    "n_estimators = range(50, 500, 50)\n",
    "max_depth=range(3,10,1)\n",
    "param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)\n",
    "kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)\n",
    "grid_search = GridSearchCV(model, param_grid, scoring=\"roc_auc\", n_jobs=-1, cv=kfold)\n",
    "grid_result = grid_search.fit(X, y)\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best: 0.735478 using {'n_neighbors': 6}\n"
     ]
    }
   ],
   "source": [
    "#tune knn\n",
    "#tune number of neigbors\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "model =KNeighborsClassifier()\n",
    "n_neighbors = range(5, 20, 1)\n",
    "param_grid = dict(n_neighbors=n_neighbors)\n",
    "kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)\n",
    "grid_search = GridSearchCV(model, param_grid, scoring=\"roc_auc\", n_jobs=-1, cv=kfold)\n",
    "grid_result = grid_search.fit(X, y)\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best: 0.676890 using {'penalty': 'l1'}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#tune logistic regression\n",
    "#tune l1 or l2 regularization\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "model =LogisticRegression()\n",
    "penalty=('l1','l2')\n",
    "param_grid = dict(penalty=penalty)\n",
    "kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)\n",
    "grid_search = GridSearchCV(model, param_grid, scoring=\"roc_auc\", n_jobs=-1, cv=kfold)\n",
    "grid_result = grid_search.fit(X, y)\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    "#based on the cross validation result, the best model is as follow:\n",
    "model=ensemble.RandomForestClassifier(max_depth=19,max_features='log2',n_estimators=450)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Precision-recall trade off: find the best threshold\n",
    "1. In this problem, we want to capture the delays as much as possible. False positives are less important than false negatives. Since recall is more important than precision, we use an adjusted F1 score as metric, which gives more weight to recall, to find the best threshold (no accurate meaning, need more business data to have more specific metrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>model</th>\n",
       "      <th>threshold</th>\n",
       "      <th>recall</th>\n",
       "      <th>precision</th>\n",
       "      <th>metrics_defined</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.15</td>\n",
       "      <td>0.443169</td>\n",
       "      <td>0.484511</td>\n",
       "      <td>0.304095</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.10</td>\n",
       "      <td>0.613746</td>\n",
       "      <td>0.271640</td>\n",
       "      <td>0.288184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.20</td>\n",
       "      <td>0.352756</td>\n",
       "      <td>0.650777</td>\n",
       "      <td>0.277536</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.295883</td>\n",
       "      <td>0.736249</td>\n",
       "      <td>0.246376</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.30</td>\n",
       "      <td>0.241959</td>\n",
       "      <td>0.807950</td>\n",
       "      <td>0.210447</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.35</td>\n",
       "      <td>0.188018</td>\n",
       "      <td>0.889056</td>\n",
       "      <td>0.170039</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.40</td>\n",
       "      <td>0.148664</td>\n",
       "      <td>0.900735</td>\n",
       "      <td>0.137331</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.45</td>\n",
       "      <td>0.126810</td>\n",
       "      <td>0.942369</td>\n",
       "      <td>0.118816</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           model  threshold    recall  precision  metrics_defined\n",
       "1  Random Forest       0.15  0.443169   0.484511         0.304095\n",
       "0  Random Forest       0.10  0.613746   0.271640         0.288184\n",
       "2  Random Forest       0.20  0.352756   0.650777         0.277536\n",
       "3  Random Forest       0.25  0.295883   0.736249         0.246376\n",
       "4  Random Forest       0.30  0.241959   0.807950         0.210447\n",
       "5  Random Forest       0.35  0.188018   0.889056         0.170039\n",
       "6  Random Forest       0.40  0.148664   0.900735         0.137331\n",
       "7  Random Forest       0.45  0.126810   0.942369         0.118816"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#precision recall trade off:\n",
    "# \n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "Performance=[]\n",
    "for thr in np.arange(0.1,0.5,0.05):\n",
    "    re=[]\n",
    "    pr=[]\n",
    "    kf = StratifiedKFold(n_splits=4, random_state=42)\n",
    "    for train_index, test_index in kf.split(X.values,y.values):\n",
    "        X_train_folds = X.values[train_index]\n",
    "        y_train_folds = (y.values[train_index])\n",
    "        X_test_fold = X.values[test_index]\n",
    "        y_test_fold = (y.values[test_index])\n",
    "        model.fit(X_train_folds, y_train_folds)\n",
    "        y_predict=pd.DataFrame(model.predict_proba(X_test_fold))\n",
    "        b=y_predict.iloc[:,0]\n",
    "        y_predict['prediction']=np.where(b>=thr,0,1)\n",
    "        y_predict=y_predict['prediction'].values\n",
    "        m=confusion_matrix(y_test_fold,y_predict)\n",
    "        re.append(m[0,0]/(m[0,0]+m[0,1]))\n",
    "        pr.append(m[0,0]/(m[0,0]+m[1,0]))\n",
    "    PR=sum(pr)/len(pr)\n",
    "    RE=sum(re)/len(re)\n",
    "    performance=['Random Forest',thr,RE,PR,2/(2/RE+1/PR)]\n",
    "    Performance.append(performance)\n",
    "Performance=pd.DataFrame(Performance)\n",
    "Performance.columns=['model','threshold','recall','precision','metrics_defined']\n",
    "Performance.sort_values(by='metrics_defined',ascending=False)\n",
    "\n",
    "#Based on the metric, the best threshold is 0.15\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test result: accuracy 91.518% recall 37.427% precision 44.755%\n"
     ]
    }
   ],
   "source": [
    "#test on test data now\n",
    "\n",
    "X_test=df_test_model.drop('ontime_ornot',axis=1)\n",
    "y_test=df_test_model['ontime_ornot']\n",
    "model.fit(X,y)\n",
    "y_predict=pd.DataFrame(model.predict_proba(X_test))\n",
    "b=y_predict.iloc[:,0]\n",
    "y_predict['prediction']=np.where(b>=0.15,0,1)\n",
    "m=confusion_matrix(y_test,y_predict['prediction'])\n",
    "re=m[0,0]/(m[0,0]+m[0,1])\n",
    "pr=m[0,0]/(m[0,0]+m[1,0])\n",
    "ac=(m[0,0]+m[1,1])/(m[0,0]+m[1,1]+m[0,1]+m[1,0])\n",
    "\n",
    "print('test result: accuracy','{:.3%}'.format(ac),'recall','{:.3%}'.format(re),'precision','{:.3%}'.format(pr))\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
