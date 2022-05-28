import matplotlib.pyplot as plt
import Fcs

Trax=[5,8,9,5,-3]
pos=[0,2.5,4,4.5,5.08]
x_deriv=[0]
y_deriv=[0]
sum_ave=Fcs.bd_Deriv_kriging_func(pos,Trax,x_deriv,y_deriv,'cst','cub',100,0)

plt.ylabel('y')
plt.xlabel('x') 
plt.legend(loc='upper left', ncol=1)  
plt.savefig('s.png',dpi=300)
