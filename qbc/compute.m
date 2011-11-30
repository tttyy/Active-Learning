function [] = compute()

err = zeros(100,1);

for i = 1:20
    j = i * 100;
    x = (-0.67) * j;
    x = x / 60000;
    err(i) = 50 * (2^x);
    err(i) = 1 - (err(i)/100);
    disp(['[' num2str(j) ',' num2str(err(i)) '],']);
end

%dlmwrite('theoryerror.txt', err, 'precision', '%.4f');

end