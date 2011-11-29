function [] = generate(len)

cnt = 0;

res = rand(len, 10);
res0 = rand(len, 11);

res = 2000000 * res - 1000000;
w = [1 1 1 1 1 0 0 0 0 5];

for i = 1:len
    k = w * res(i,:)';
    %k = sum(res(i,:));
    res0(i,1:10) = res(i,:);
    res0(i,11) = -1;
    if(k > 0)
        cnt = cnt + 1;
        res0(i,11) = 1;
    end
end

dlmwrite('train.txt', res0, 'precision', '%.0f');

disp(cnt);
disp(len - cnt);

end