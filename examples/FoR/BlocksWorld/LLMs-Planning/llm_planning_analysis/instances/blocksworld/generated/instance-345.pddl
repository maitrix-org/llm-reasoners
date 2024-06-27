(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b l d i)
(:init 
(handempty)
(ontable b)
(ontable l)
(ontable d)
(ontable i)
(clear b)
(clear l)
(clear d)
(clear i)
)
(:goal
(and
(on b l)
(on l d)
(on d i)
)))