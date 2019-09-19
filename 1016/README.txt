# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:11:13 2018

@author: Akitaka
"""

[1a3] y-randamization でcos, sinをやったばあい、kNN, RF, SVMではグラフがどうなる？
ref 水素化物のでもy-randomization (20181009,11,12)

結果
SVR: データの真ん中あたりを通る曲線
　　極端に不自然なグラフではないが、再現はできていない(できなくていい)
RF, kNN: ぐにゃぐにゃ曲がるグラフになる
　　そもそもが脈絡のないデータなので、むりやり再現しようとすれば当然

