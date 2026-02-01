  静态特征（构建 X_pro 矩阵）：                                                                                                             
  pro_total_seasons      # 参加过的总赛季数                                                                                                 
  pro_total_celebs       # 配对过的名人总数                                                                                                 
  pro_best_placement     # 历史最好名次                                                                                                     
  pro_win_rate           # 历史胜率（冠军次数/参赛次数）                                                                                    
  pro_avg_weeks          # 平均每季存活周数                                                                                                 
  pro_experience_years   # 从业年数（如果有数据）                                                                                           
                                                                                                                                            
  动态特征（放在 X_obs 中）：                                                                                                               
  pro_current_streak     # 当前连续获胜/失败周数                                                                                            
  pro_season_rank        # 本赛季当前排名                                                                                                   
  pro_partner_chemistry  # 与当前名人的"化学反应"（可用历史数据估计）