function [] = generate(len)

cnt = 0;

res = rand(len, 10);
res0 = rand(len, 12);

res = 2000000 * res - 1000000;


for i = 1:len
    k = sum(res(i,:));
    res0(i,1:10) = res(i,:);
    res0(i,11) = -1;
    res0(i,12) = -1;
    if(k > 0)
        cnt = cnt + 1;
        res0(i,12) = 1;
    end
end

dlmwrite('test.txt', res0, 'precision', '%.0f');

disp(cnt);
disp(len - cnt);

end