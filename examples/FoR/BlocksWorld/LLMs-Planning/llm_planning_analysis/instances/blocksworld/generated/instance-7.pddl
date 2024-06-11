(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g e d j)
(:init 
(handempty)
(ontable g)
(ontable e)
(ontable d)
(ontable j)
(clear g)
(clear e)
(clear d)
(clear j)
)
(:goal
(and
(on g e)
(on e d)
(on d j)
)))