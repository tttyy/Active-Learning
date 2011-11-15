function [] = generate()

len = 10000;
res = zeros(len, 2);
cnt = 0;

x = rand(1, len);
y = rand(1, len);

x = 200000 * x - 100000;
y = 200000 * y - 100000;

for i = 1:len
    res(i, :) = [x(i) y(i)];
    k = x(i) + y(i);
    if(k > 0)
        cnt = cnt + 1;
    end
end

dlmwrite('train.txt', res, 'precision', '%.0f');

disp(cnt);

end