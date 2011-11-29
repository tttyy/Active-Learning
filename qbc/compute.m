function [] = compute()

err = zeros(100,1);

for i = 1:100
    x = (-0.67) * i * 10;
    x = x / 100;
    err(i) = 50 * (2^x);
    disp([i*10 err(i)]);
end

%dlmwrite('theoryerror.txt', err, 'precision', '%.4f');

end