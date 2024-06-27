(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f e i c)
(:init 
(handempty)
(ontable f)
(ontable e)
(ontable i)
(ontable c)
(clear f)
(clear e)
(clear i)
(clear c)
)
(:goal
(and
(on f e)
(on e i)
(on i c)
)))