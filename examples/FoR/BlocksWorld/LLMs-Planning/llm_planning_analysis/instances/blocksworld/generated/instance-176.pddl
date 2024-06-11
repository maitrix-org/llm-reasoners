(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d e l)
(:init 
(handempty)
(ontable d)
(ontable e)
(ontable l)
(clear d)
(clear e)
(clear l)
)
(:goal
(and
(on d e)
(on e l)
)))