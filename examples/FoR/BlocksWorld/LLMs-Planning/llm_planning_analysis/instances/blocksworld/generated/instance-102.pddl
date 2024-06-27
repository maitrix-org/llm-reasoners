(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l e c)
(:init 
(handempty)
(ontable l)
(ontable e)
(ontable c)
(clear l)
(clear e)
(clear c)
)
(:goal
(and
(on l e)
(on e c)
)))