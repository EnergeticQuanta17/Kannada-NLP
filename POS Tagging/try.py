def get_function(n):
    if(n==1):
        def f1(n1):
            return (n1+1)
        return f1
    
    if(n==2):
        def f2(n1):
            return (n1+2)
        
        return f2

print(get_function(2)(2))