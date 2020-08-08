using Plots
using ProgressMeter
using JLD
using LinearAlgebra
using CSV

function calForce(dist0, k, force, type)     # tVal is not used 
    function fn(tVal, xList, vList) 
        if(size(xList, 2) != size(vList, 2))
            println("Error: length of xList and vLsit are different (at func)")
            return
        end
        leng =  size(xList, 2)
        if(leng != (length(dist0)+1))
            println("Error: length and dist0 are different (at func)")
            return
        end
        acc = zeros(Float64, 2, leng)
        val = [force(k, dist0[i], xList[:, i], xList[:, i+1]) for i in 1:(leng-1)]
        acc[:, 2:leng-1] = [ -val[i-1][j] + val[i][j] for i in 2:leng-1 for j in 1:2]
        acc[:, 1] = (type != "fix") * val[1]
        acc[:,leng] = -(type != "fix") * val[leng-1]
        return acc
    end
    return fn

end

function spring(k, l0, x0, x1)
    x = x1-x0
    l = norm(x)
    u = x/l
    f = k * (l-l0) * u
    return f
end

function molecule(e, l0, x0, x1) #  Lennard–Jones interaction potential
    x = norm(x1-x0)
    u = (x1-x0)/x
    s = (l0^6)^(1/6)/2^(1/6)
    return (24*e*(s^6)*(-2*s^6 + x^6))/x^13 * u
end

function rk4(f, t, x, v)
        kx1 = v
        kv1 = f( t , x, v )
    
        kx2 = v + h*kv1/2
        kv2 = f( t + h/2, x + h*kx1/2, v + h*kv1/2 )
    
        kx3 = v + h*kv2/2
        kv3 = f( t + h/2, x + h*kx2/2, v + h*kv2/2 )

        kx4 = v + h*kv3
        kv4 = f( t + h, x + h*kx3, v + h*kv3 )
    
        dx = h*(kx1 + 2*kx2 + 2*kx3 + kx4)/6
        dv = h*(kv1 + 2*kv2 + 2*kv3 + kv4)/6
    
        x = x + dx
        v = v + dv
    
    return (x, v);
end

h = 0.1
t = 0:h:8000
leng = 701
num = 10

x = zeros(Float64, 2, leng, length(t), num)     # postion, num, time, num(initial velocity)    
v = zeros(Float64, 2, leng, length(t), num)   
record = zeros(Int, leng, num)

x[1, :, 1, :] = repeat(collect(1:leng), num)
v[2, trunc(Int, (leng+1)/2), 1, :] = collect(range(0.03, 0.3, length = num))

xDistance = zeros(Float64, leng-1, num)
for i in 1:num
    xDistance[:, i] = [norm(x[:,j+1,1,i] - x[:,j,1,i]) for j in 1:(leng-1) ]
end

p = Progress(num*(length(t)-1))
Threads.@threads for i in 1:num
    for j in 2:length(t)
        (x[:,:,j,i], v[:,:,j,i]) = rk4(calForce(xDistance[:,1], 0.1, molecule, "fix"), 0, x[:,:,j-1,i], v[:,:,j-1,i])
        next!(p)    
    end
end

#= 
Threads.@threads for i in 1:num
    for j in 2:length(t)    
        (x[:,:,j,i], v[:,:,j,i]) = rk4(calForce(xDistance[:,1], 8.1, spring, "fix"), 0, x[:,:,j-1,i], v[:,:,j-1,i])
        next!(p)    
    end
end
=#

plot(t, x[2, trunc(Int, (leng+1)/2), :, 1])
plot(x[1, :, 20000, 1], x[2, :, 20000, 1])
for i in 1:num
    for j in 1:leng
        a= findfirst(x->x!=0, x[2, j, :, num])
        if(a === nothing) 
            record[j, i] = 0
        else 
            record[j, i] = a
        end
    end
end

save("C:\\Users\\USER\\Documents\\linewave\\result\\molecule-data.jld", "t", t, "x", x, "v", v, "record", record)

plot(record[351:400, :], xlabel = "pos", ylabel = "time")
savefig("C:\\Users\\USER\\Documents\\linewave\\result\\molecule-record.png")

plot( x[1, :, 400, 10], x[2, :, 400, 10] )

p2 = Progress(num*(length(t)-1))
for i in 1:50:length(t)
    plot(x[1, :, i, 1], x[2, :, i, 1], title = "molecule, time"*string(i*h), label = "v = 0.0001, n=701", ylims = (-0.01,0.05))
    savefig("C:\\Users\\USER\\Documents\\linewave\\gif2\\"*string((i-1)/50)*".png")
    next!(p2)
end

