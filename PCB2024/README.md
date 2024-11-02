
# Main Source File　
## Fig1  
1\_cv\_a.c : Calculation with different values of a.  
1\_cv\_b.c : Calculation with different values of b.  
1\_cv\_k.c : Calculation with different values of k. 

## Fig2　  
1\_cv\_a.c : Calculation with different values of a  
1\_cv\_b.c : Calculation with different values of b  
1\_cv\_f.c : Calculation with different values of f
1\_cv\_k.c : Calculation with different values of k

## Fig3　  
1\_biggs.c : Gibbs sampling with numerical CV  

## Fig4　  
1\_cv\_ana.c : Calculation of CV with analytical formula  
1\_cv\_num.c : Numerical calculation of CV with checkpoints

## Fig5　 
1\_gibbs.c : Gibbs sampling with analytical formula  

## FigS1　 
1\_cv\_u\_a.c : Numerical calculation of CV for u with different values a   
1\_cv\_u\_b.c : Numerical calculation of CV for u with different values b   
1\_cv\_u\_k.c : Numerical calculation of CV for u with different values k   
1\_cv\_v\_a.c : Numerical calculation of CV for v with different values a   
1\_cv\_v\_b.c : Numerical calculation of CV for v with different values b   
1\_cv\_v\_k.c : Numerical calculation of CV for v with different values k   
1\_cv\_w\_a.c : Numerical calculation of CV for w with different values a   
1\_cv\_w\_b.c : Numerical calculation of CV for w with different values b   
1\_cv\_w\_k.c : Numerical calculation of CV for w with different values k  

## FigS2　 
1\_de.c : Differential evolution of numerical CV  

## FigS3　  
1_de.c : Differential evolution of analytical CV  

# To compile the C code including Mersenne Twister random Method　
```
Gcc-14 -fopenmp -Ofast -msse2 -DSFMT_MEXP=19937 SFMT.c “Main Source File” func.c -lm
```
