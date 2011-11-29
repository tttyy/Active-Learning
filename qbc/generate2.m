function [] = generate2()

len = 50000;
cnt = 1;
res = rand(len, 20);
res0 = randperm(1000000);
value = zeros(1, len);

for i = 1:1000000
    k = rand();
    if(k < 0.06)
        res(cnt,:) = de2bi(res0(i), 20);
        cnt = cnt + 1;
    end
    if(cnt > len)
        break;
    end
end

pos = 0;
w = [1 1 1 0 1 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0];
%for i = 1:20
%    pos = pos + w(i) * w(i);
%end
%pos = sqrt(pos);
%w = w / pos;

for i = 1:len
    value(i) = w * res(i,:)';
end

dlmwrite('train_binary2.txt', res, 'precision', '%.0f');


end