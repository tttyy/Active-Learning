function [] = generate()

len = 10000;

cnt = 0;

%res = rand(len, 10);
res = rand(len, 3);

res = 2000000 * res - 1000000;


for i = 1:len
    k = sum(res(i,:));
    if(k > 0)
        cnt = cnt + 1;
    end
end

dlmwrite('train2.txt', res, 'precision', '%.0f');

disp(cnt);
disp(len - cnt);

end