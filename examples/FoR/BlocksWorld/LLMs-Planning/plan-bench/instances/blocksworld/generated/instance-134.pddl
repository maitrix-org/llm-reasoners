(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l e c)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable e)
(ontable c)
(clear d)
(clear l)
(clear e)
(clear c)
)
(:goal
(and
(on d l)
(on l e)
(on e c)
)))