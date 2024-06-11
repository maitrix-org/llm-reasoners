(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b d i)
(:init 
(handempty)
(ontable b)
(ontable d)
(ontable i)
(clear b)
(clear d)
(clear i)
)
(:goal
(and
(on b d)
(on d i)
)))