(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d e c g)
(:init 
(handempty)
(ontable d)
(ontable e)
(ontable c)
(ontable g)
(clear d)
(clear e)
(clear c)
(clear g)
)
(:goal
(and
(on d e)
(on e c)
(on c g)
)))