module HyperRPNNs

using LinearAlgebra
using Quaternions
using Random
using Statistics

rng = MersenneTwister(1234);

            
### Bilinear form used in the paper!

function LambdaInner(U,x,Params)
    w = zeros(size(U,3),1)  
    for i = 1:size(U,2)
        w = w + Params[i]*(U[:,i,:]'*x[:,i])
    end        
    return w
end

#function train(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U,alpha, beta)
function train(BilinearForm, BilinearFormParams, U,alpha, beta)    
    N = size(U,1)
    hyperN = size(U,2)
    P = size(U,3)
    
    C       = zeros(P,P)
    
    for i=1:P
        
        C[:,i] = exp.(alpha*(BilinearForm(U,U[:,:,i],BilinearFormParams)/N) .+ beta);
        
    end
    
    return inv(C)
end

### Some activation functions

function csign(x,K)
    z = x[:,1]+x[:,2]*im
    phase_quanta = (round.(K*(2*pi.+angle.(z))./(2*pi))).%K
    z = exp.(2.0*pi*phase_quanta*im/K)
    return hcat(real.(z),imag.(z))
end

function twincsign(x, Params = 16)
    c1 = csign(x[:,1:2],Params)
    c2 = csign(x[:,3:4],Params)
    return hcat(c1,c2)
end

function SplitSign(a, Params = nothing)
    x = zeros(size(a))
    for i = 1:size(a,3)
        x[:,:,i] = sign.(a[:,:,i])
    end
    return x
end


##########################################################################
### Generic Hypercomplex- Valued -Recurrent-Projection Networks
##########################################################################

function Sync(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max =1.e3)
    ### Exponential Hypercomplex RPNN
    Name = "Hypercomplex RPNN (Synchronous)"
   

    N = size(U,1)
    hyperN = size(U,2)
    P = size(U,3)
    tau = 1.e-4
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau   


    # Storing the first element of the sequence
    Sequence = []
    append!(Sequence,[x])
    
    a = zeros(N,hyperN)
    
    # The parameters of the function are: 
    #Cinv = train(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, alpha, beta)
    Cinv = train(BilinearForm, BilinearFormParams, U, alpha, beta)
    
    while (Error>tau)&&(it<it_max)
        it = it+1
        
        # Compute the weights;
        w = exp.(alpha*(BilinearForm(U,x,BilinearFormParams)/N) .+ beta);
                    
        g = Cinv*w
        
        # Compute the quaternion-valued activation potentials;
        for j = 1:hyperN
            a[:,j] = U[:,j,:]*g
        end

        # Compute the next state;
        x = ActFunction(a, ActFunctionParams)

        Error = norm(x-xold)
        xold = copy(x)

    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, it
end

function Seq(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
    ### Exponential Hypercomplex RPNN
    Name = "Hypercomplex RPNN (Asynchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    Energy = zeros(1)
    
    a = zeros(1,hyperN)
    
    # The parameters of the function are: 
    #Cinv = train(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, alpha, beta)
    Cinv = train(BilinearForm, BilinearFormParams, U, alpha, beta)
    
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            
            # Compute the quaternion-valued activation potentials;
            w = exp.(alpha*(BilinearForm(U,x,BilinearFormParams)/(N)) .+ beta);
            
            g = Cinv*w
            
            for j = 1:hyperN
                a[1,j] = dot(U[ind[i],j,:],g)
            end

            # Compute the next state;
            x[ind[i],:] = ActFunction(a, ActFunctionParams)

        end
        
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end

function Seq_slow(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
    ### Exponential Hypercomplex RPNN
    Name = "Hypercomplex RPNN (Asynchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    Energy = zeros(1)
    a = zeros(1,hyperN)

    # The parameters of the function are: 
    # Cinv = train(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, alpha, beta)
    Cinv = train(BilinearForm, BilinearFormParams, U, alpha, beta)
    
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            
            w = exp.(alpha*(BilinearForm(U,x,BilinearFormParams)/(N)) .+ beta);
            g = Cinv*w
            
            # Compute the quaternion-valued activation potentials;
            for j = 1:hyperN
                a[1,j] = dot(U[ind[i],j,:],g)
            end

            # Compute the next state;
            x[ind[i],:] = ActFunction(a, ActFunctionParams)
        end
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end


##########################################################################
### Unit Quaternion Networks
##########################################################################
function UnitQ_Sync(U, xinput, alpha = 1, beta=0, it_max = 1.e3)
    ### Exponential Unit Quaternion RPNN
    Name = "Unit Quaternion RPNN (Synchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau
    
    BilinearForm = LambdaInner
    BilinearFormParams = [hyperN]

    # Storing the first element of the sequence
    Sequence = []
    append!(Sequence,[x])

    Cinv = train(BilinearForm, BilinearFormParams, U, alpha, beta)
    
    a = zeros(N,hyperN)

                
    while (Error>tau)&&(it<it_max)
        it = it+1
        
        w = exp.(alpha*Array{Float64}(BilinearForm(U,x,BilinearFormParams)/N) .+ beta);
        
        # Compute the new weights
        g = Cinv*w
        
        
        # Compute the quaternion-valued activation potentials;
        for j = 1:hyperN
            a[:,j] = U[:,j,:]*g
        end
        
        ###
        x = a ./ abs.(a)
        
        Error = norm(x-xold)
        xold = copy(x)
        append!(Sequence,[xold])
        
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.")  
    end
    return x, Sequence
end

function UnitQ_Seq(U, xinput, alpha = 1, beta=0, it_max = 1.e3)
    ### Exponential Unit Quaternion RPNN
    Name = "Unit Quaternion RPNN (Asynchronous)"

    N = size(U,1)
    tau = 1.e-6

    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    
    # Compute the energy
    Energy = zeros(Float64,(1,))
    
    Cinv = train(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, alpha, beta)
    
    V = U*Cinv;
                              
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            
            w = exp.(alpha*Array{Float64}(real(U'*x)/(L*N)).+beta);
            
            # Compute the quaternion-valued activation potentials;
            a = dot(conj(V[ind[i],:]),w)

            # Compute the next state;
            x[ind[i]] = a/abs(a)
        end
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end

end
