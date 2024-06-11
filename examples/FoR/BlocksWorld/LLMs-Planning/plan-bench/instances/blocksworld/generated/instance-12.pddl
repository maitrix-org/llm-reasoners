(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i k l f)
(:init 
(handempty)
(ontable i)
(ontable k)
(ontable l)
(ontable f)
(clear i)
(clear k)
(clear l)
(clear f)
)
(:goal
(and
(on i k)
(on k l)
(on l f)
)))