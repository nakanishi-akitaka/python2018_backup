# -*- coding: utf-8 -*-
"""
Pandas でデータフレームの結合 (マージ, JOIN)
のページでは、Pandas で作成したデータフレーム同士を結合する方法について紹介します。
Created on Wed Jun 27 13:30:30 2018

@author: Akitaka
"""

# 2 つのデータフレームを結合する
# 以下の例では、merge() メソッドを用いて、2 つのデータフレームを作成し、
# 内部結合 (inner join) を行います。

import pandas as pd
import numpy as np
 
# データフレーム customer (顧客) を作成
customer = pd.DataFrame([["0001", "John"], ["0002", "Lily"]], columns=['customer_id', 'name'])
print(customer)

# データフレーム order (注文) を作成
order = pd.DataFrame([["0001", "Smartphone"],
                      ["0001", "Wireless Charger"],
                      ["0002", "Wearable watch"]],
                      columns=['customer_id', 'product_name'])
print(order)

# データフレーム customer と order を内部結合
temp = pd.merge(customer, order, how="inner", on="customer_id")
print(temp)

# 結合に用いるキーが異なる場合は、left_on, right_on 引数で指定します。
# データフレーム employee (従業員) を作成
employee = pd.DataFrame([["Miki", "Tokyo"],["Ichiro", "Osaka"]],
                       columns=['employee_name', 'office_name'])
print(employee)

# データフレーム office (事務所) を作成
office = pd.DataFrame([["Tokyo", "1-2-3 Chiyoda-ku Tokyo"],
                       ["Osaka", "3-2-1 Chuo-ku Osaka"]],
                      columns=['name', 'address'])
print(office)

# データフレーム employee と office を内部結合
temp = pd.merge(employee, office, how="inner", left_on="office_name", right_on="name")
print(temp)

#%% 
# 外部結合 (Left join, outer join) を行う際は how 引数にてそれぞれ指定します。

# データフレーム products (商品) を作成
products = pd.DataFrame([["P-001", "Orange"],
                         ["P-002", "Apple"],
                         ["P-003", "Blueberry"]],
                        columns=['product_id', 'name'])
print(products)
 
# データフレーム stock (在庫) を作成
stock = pd.DataFrame([["P-001", 10],
                      ["P-002", 20],
                      ["P-010", 30]],
                     columns=['product_id', 'amount'])
print(stock)
 
# Left Join (Left outer join, 左部分外部結合)
# 右側の内、左側のidにあるものを足す。存在しないデータはNaNで埋め合わせる
temp = pd.merge(products, stock, how="left", on="product_id")
print(temp)

# Right Join (Left outer join, 右部分外部結合)
# 左側の内、右側のidにあるものを足す。存在しないデータはNaNで埋め合わせる
temp = pd.merge(products, stock, how="right", on="product_id")
print(temp)

# Outer join (Full outer join, 完全外部結合)
# 両方のidにあるものを足す。存在しないデータはNaNで埋め合わせる
temp = pd.merge(products, stock, how="outer", on="product_id")
print(temp)

#%%
# 2 つのデータフレームを結合する (concat)
# 2 つのデータフレームを縦方向に結合するには、pd.concat メソッドを用いて行えます。
df1 = pd.DataFrame([["0001", "John"],
                    ["0002", "Lily"]],
                   columns=['id', 'name'])
print(df1) 
df2 = pd.DataFrame([["0003", "Chris"],
                    ["0004", "Jessica"]],
                   columns=['id', 'name'])
print(df2)
temp = pd.concat([df1, df2], ignore_index=True)
print(temp)
