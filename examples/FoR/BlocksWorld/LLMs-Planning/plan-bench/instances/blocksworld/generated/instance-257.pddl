(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l d g i)
(:init 
(handempty)
(ontable l)
(ontable d)
(ontable g)
(ontable i)
(clear l)
(clear d)
(clear g)
(clear i)
)
(:goal
(and
(on l d)
(on d g)
(on g i)
)))