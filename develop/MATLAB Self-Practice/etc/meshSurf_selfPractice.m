%Playing with surface and meshgrid
%The purpose of this code is to get myself comfortable with meshgrid and
%surf function


a = linspace(1,10,8);
b = linspace(2,10,5);

[A, B] = meshgrid(a, b);
Z2 = A .*B;

for i = 1:length(b)
    for j = i:length(a)
        Z1(i,j) = b(i)*a(j);
    end
end

%{
figure(1);
surf(a,b,Z); %a and b are vector arguments
hold on;
xlabel('x');
ylabel('y');
hold off;
%}

%{
figure(2);
surf(A,B,Z); %A and B are matrix arguments
xlabel('x');
ylabel('y');
%}