\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  min_dist[1] + min_dist[3] + min_dist[4] + min_dist[2] + 0 xfc_dists[1,3]
   + 0 xfc_dists[1,4] + 0 xfc_dists[1,2] + 0 xfc_dists[3,1]
   + 0 xfc_dists[3,4] + 0 xfc_dists[3,2] + 0 xfc_dists[4,1]
   + 0 xfc_dists[4,3] + 0 xfc_dists[4,2] + 0 xfc_dists[2,1]
   + 0 xfc_dists[2,3] + 0 xfc_dists[2,4]
Subject To
 num_xfc_constraint: is_xfc[1] + is_xfc[3] + is_xfc[4] + is_xfc[2] = 5
 self_dist_constraint_1_1: xfc_dists[1,1] = 0
 self_dist_constraint_3_3: xfc_dists[3,3] = 0
 self_dist_constraint_4_4: xfc_dists[4,4] = 0
 self_dist_constraint_2_2: xfc_dists[2,2] = 0
 xfc_dist_constraint_1_3: is_xfc[3] = 1 -> xfc_dists[1,3] = 1e-08
 non_xfc_dist_constraint_1_3: is_xfc[3] = 0 -> xfc_dists[1,3] = 1e+100
 xfc_dist_constraint_1_4: is_xfc[4] = 1 -> xfc_dists[1,4] = 10.00000001
 non_xfc_dist_constraint_1_4: is_xfc[4] = 0 -> xfc_dists[1,4] = 1e+100
 xfc_dist_constraint_1_2: is_xfc[2] = 1 -> xfc_dists[1,2] = 10.00000002
 non_xfc_dist_constraint_1_2: is_xfc[2] = 0 -> xfc_dists[1,2] = 1e+100
 xfc_dist_constraint_3_1: is_xfc[1] = 1 -> xfc_dists[3,1] = inf
 non_xfc_dist_constraint_3_1: is_xfc[1] = 0 -> xfc_dists[3,1] = 1e+100
 xfc_dist_constraint_3_4: is_xfc[4] = 1 -> xfc_dists[3,4] = 10
 non_xfc_dist_constraint_3_4: is_xfc[4] = 0 -> xfc_dists[3,4] = 1e+100
 xfc_dist_constraint_3_2: is_xfc[2] = 1 -> xfc_dists[3,2] = 10.00000001
 non_xfc_dist_constraint_3_2: is_xfc[2] = 0 -> xfc_dists[3,2] = 1e+100
 xfc_dist_constraint_4_1: is_xfc[1] = 1 -> xfc_dists[4,1] = inf
 non_xfc_dist_constraint_4_1: is_xfc[1] = 0 -> xfc_dists[4,1] = 1e+100
 xfc_dist_constraint_4_3: is_xfc[3] = 1 -> xfc_dists[4,3] = inf
 non_xfc_dist_constraint_4_3: is_xfc[3] = 0 -> xfc_dists[4,3] = 1e+100
 xfc_dist_constraint_4_2: is_xfc[2] = 1 -> xfc_dists[4,2] = 1e-08
 non_xfc_dist_constraint_4_2: is_xfc[2] = 0 -> xfc_dists[4,2] = 1e+100
 xfc_dist_constraint_2_1: is_xfc[1] = 1 -> xfc_dists[2,1] = inf
 non_xfc_dist_constraint_2_1: is_xfc[1] = 0 -> xfc_dists[2,1] = 1e+100
 xfc_dist_constraint_2_3: is_xfc[3] = 1 -> xfc_dists[2,3] = inf
 non_xfc_dist_constraint_2_3: is_xfc[3] = 0 -> xfc_dists[2,3] = 1e+100
 xfc_dist_constraint_2_4: is_xfc[4] = 1 -> xfc_dists[2,4] = inf
 non_xfc_dist_constraint_2_4: is_xfc[4] = 0 -> xfc_dists[2,4] = 1e+100
Bounds
Binaries
 is_xfc[1] is_xfc[3] is_xfc[4] is_xfc[2]
General Constraints
 min_dist_constraint_Node(1): min_dist[1] = MIN ( xfc_dists[1,1] ,
   xfc_dists[1,3] , xfc_dists[1,4] , xfc_dists[1,2] )
 min_dist_constraint_Node(3): min_dist[3] = MIN ( xfc_dists[3,1] ,
   xfc_dists[3,3] , xfc_dists[3,4] , xfc_dists[3,2] )
 min_dist_constraint_Node(4): min_dist[4] = MIN ( xfc_dists[4,1] ,
   xfc_dists[4,3] , xfc_dists[4,4] , xfc_dists[4,2] )
 min_dist_constraint_Node(2): min_dist[2] = MIN ( xfc_dists[2,1] ,
   xfc_dists[2,3] , xfc_dists[2,4] , xfc_dists[2,2] )
End
