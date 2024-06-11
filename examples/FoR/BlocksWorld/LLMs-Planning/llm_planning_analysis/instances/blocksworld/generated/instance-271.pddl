(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i c j g)
(:init 
(handempty)
(ontable i)
(ontable c)
(ontable j)
(ontable g)
(clear i)
(clear c)
(clear j)
(clear g)
)
(:goal
(and
(on i c)
(on c j)
(on j g)
)))