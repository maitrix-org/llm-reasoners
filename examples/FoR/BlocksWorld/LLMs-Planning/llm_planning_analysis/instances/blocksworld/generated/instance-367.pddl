(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c e l)
(:init 
(handempty)
(ontable c)
(ontable e)
(ontable l)
(clear c)
(clear e)
(clear l)
)
(:goal
(and
(on c e)
(on e l)
)))