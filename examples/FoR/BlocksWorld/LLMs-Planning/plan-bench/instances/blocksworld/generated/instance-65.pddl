(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d k e)
(:init 
(handempty)
(ontable d)
(ontable k)
(ontable e)
(clear d)
(clear k)
(clear e)
)
(:goal
(and
(on d k)
(on k e)
)))