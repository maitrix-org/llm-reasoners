(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d a e)
(:init 
(handempty)
(ontable d)
(ontable a)
(ontable e)
(clear d)
(clear a)
(clear e)
)
(:goal
(and
(on d a)
(on a e)
)))