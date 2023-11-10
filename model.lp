\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  min_Node(1) + min_Node(3) + min_Node(4) + min_Node(2)
Subject To
 num_xfc_constraint: is_xfc[1] + is_xfc[3] + is_xfc[4] + is_xfc[2] = 2
 dist_constraint_Node(1)_Node(1): dist_Node(1)_Node(1) = 0
 xfc_constraint_Node(1)_Node(1): - 10.00000002 is_xfc[1]
   - dist_Node(1)_Node(1) + xfc_Node(1)_Node(1) = 0
 dist_constraint_Node(1)_Node(3): dist_Node(1)_Node(3) = 1e-08
 xfc_constraint_Node(1)_Node(3): - 10.00000002 is_xfc[3]
   - dist_Node(1)_Node(3) + xfc_Node(1)_Node(3) = 0
 dist_constraint_Node(1)_Node(4): dist_Node(1)_Node(4) = 10.00000001
 xfc_constraint_Node(1)_Node(4): - 10.00000002 is_xfc[4]
   - dist_Node(1)_Node(4) + xfc_Node(1)_Node(4) = 0
 dist_constraint_Node(1)_Node(2): dist_Node(1)_Node(2) = 10.00000002
 xfc_constraint_Node(1)_Node(2): - 10.00000002 is_xfc[2]
   - dist_Node(1)_Node(2) + xfc_Node(1)_Node(2) = 0
 dist_constraint_Node(3)_Node(3): dist_Node(3)_Node(3) = 0
 xfc_constraint_Node(3)_Node(3): - 10.00000002 is_xfc[3]
   - dist_Node(3)_Node(3) + xfc_Node(3)_Node(3) = 0
 dist_constraint_Node(3)_Node(4): dist_Node(3)_Node(4) = 10
 xfc_constraint_Node(3)_Node(4): - 10.00000002 is_xfc[4]
   - dist_Node(3)_Node(4) + xfc_Node(3)_Node(4) = 0
 dist_constraint_Node(3)_Node(2): dist_Node(3)_Node(2) = 10.00000001
 xfc_constraint_Node(3)_Node(2): - 10.00000002 is_xfc[2]
   - dist_Node(3)_Node(2) + xfc_Node(3)_Node(2) = 0
 dist_constraint_Node(4)_Node(4): dist_Node(4)_Node(4) = 0
 xfc_constraint_Node(4)_Node(4): - 10.00000002 is_xfc[4]
   - dist_Node(4)_Node(4) + xfc_Node(4)_Node(4) = 0
 dist_constraint_Node(4)_Node(2): dist_Node(4)_Node(2) = 1e-08
 xfc_constraint_Node(4)_Node(2): - 10.00000002 is_xfc[2]
   - dist_Node(4)_Node(2) + xfc_Node(4)_Node(2) = 0
 dist_constraint_Node(2)_Node(2): dist_Node(2)_Node(2) = 0
 xfc_constraint_Node(2)_Node(2): - 10.00000002 is_xfc[2]
   - dist_Node(2)_Node(2) + xfc_Node(2)_Node(2) = 0
Bounds
Binaries
 is_xfc[1] is_xfc[3] is_xfc[4] is_xfc[2]
General Constraints
 min_constraint_Node(1): min_Node(1) = MIN ( xfc_Node(1)_Node(1) ,
   xfc_Node(1)_Node(3) , xfc_Node(1)_Node(4) , xfc_Node(1)_Node(2) )
 min_constraint_Node(3): min_Node(3) = MIN ( xfc_Node(3)_Node(3) ,
   xfc_Node(3)_Node(4) , xfc_Node(3)_Node(2) )
 min_constraint_Node(4): min_Node(4) = MIN ( xfc_Node(4)_Node(4) ,
   xfc_Node(4)_Node(2) )
 min_constraint_Node(2): min_Node(2) = MIN ( xfc_Node(2)_Node(2) )
End
