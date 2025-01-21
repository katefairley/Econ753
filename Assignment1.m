% Set initial guesses
x0 = [0.01 0.01]';
A0 = [1 2; 2 1];

% Define function
f = @(x) [x(1)^2+x(2)^2; x(1)+x(2)];

% Call broyden function to find root of f
root = broyden(f,x0,A0);

% Define broyden function
function [root]=broyden(f,x0,A0)
    % Set initial difference level
    tol = 1;

    while tol>1e-8
        % Calculate x^(t+1)
        x1 = x0 - inv(A0)*feval(f,x0);
        % Get difference between x^t and x^(t+1)
        d = x1 - x0;
        % Calculate A^(t+1)
        A1 = A0 + ((feval(f,x1) - feval(f,x0)-A0*d)/norm(d)^2)*d.';
        
        % Set new x^t and A^t for next iteration
        x0 = x1;
        A0 = A1;
        
        % Set tol as the minimum absolute difference between elements of x^t and x^(t+1)
        tol = min(abs(d));
    end
    root = x1;
end